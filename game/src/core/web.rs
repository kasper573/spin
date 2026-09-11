//! The browser page the client runs in: its canvas, storage, pointer lock and the script hooks
//! `spin_command(json)` and `spin_status()` that automation drives the app through.
//!
//! Storage is the browser's IndexedDB, which holds as much as its disk allows: one store of
//! text by key, read back in full once as the page opens, and written a batch at a time, each
//! batch all or nothing.
use std::cell::RefCell;

use bevy::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransactionMode};

const CANVAS: &str = "#glcanvas";
const DATABASE: &str = "spin-gravity-wheel";
const STORE: &str = "saves";

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        canvas: Some(CANVAS.to_owned()),
        ..default()
    }
}

/// Start reading back everything kept, which `storage_opened` hands over once it has arrived.
pub fn storage_open() {
    let request = web_sys::window()
        .and_then(|window| window.indexed_db().ok().flatten())
        .and_then(|factory| factory.open_with_u32(DATABASE, 1).ok());
    let Some(request) = request else {
        return opened(Vec::new());
    };
    let created = request.clone();
    request.set_onupgradeneeded(Some(
        Closure::once_into_js(move || {
            if let Ok(database) = created.result() {
                let _ = database
                    .unchecked_into::<IdbDatabase>()
                    .create_object_store(STORE);
            }
        })
        .unchecked_ref(),
    ));
    let succeeded = request.clone();
    request.set_onsuccess(Some(
        Closure::once_into_js(move || read_all(&succeeded)).unchecked_ref(),
    ));
    request.set_onerror(Some(
        Closure::once_into_js(|| opened(Vec::new())).unchecked_ref(),
    ));
}

/// Everything kept, by key, once it has been read back: handed over once, and nothing before.
pub fn storage_opened() -> Option<Vec<(String, String)>> {
    STORAGE.with(|storage| storage.borrow_mut().read.take())
}

/// Write a batch: clear everything kept first if asked, then put and delete these.
pub fn storage_write(clear: bool, puts: &[(String, String)], deletes: &[String]) {
    let transaction = STORAGE.with(|storage| {
        let database = storage.borrow().database.clone()?;
        let transaction = database
            .transaction_with_str_and_mode(STORE, IdbTransactionMode::Readwrite)
            .ok()?;
        let queued = transaction.object_store(STORE).and_then(|store| {
            if clear {
                store.clear()?;
            }
            for key in deletes {
                store.delete(&JsValue::from_str(key))?;
            }
            for (key, value) in puts {
                store.put_with_key(&JsValue::from_str(value), &JsValue::from_str(key))?;
            }
            Ok(())
        });
        if queued.is_err() {
            let _ = transaction.abort();
        }
        Some(transaction)
    });
    match transaction {
        Some(transaction) => WRITES.with(|hooks| {
            transaction.set_oncomplete(Some(hooks.committed.as_ref().unchecked_ref()));
            transaction.set_onabort(Some(hooks.failed.as_ref().unchecked_ref()));
        }),
        None => STORAGE.with(|storage| storage.borrow_mut().failed += 1),
    }
}

/// Batches written so far.
pub fn storage_committed() -> u32 {
    STORAGE.with(|storage| storage.borrow().committed)
}

/// Batches that failed to be written so far, none of which was kept in part.
pub fn storage_failed() -> u32 {
    STORAGE.with(|storage| storage.borrow().failed)
}

/// Keep bevy's window in step with the canvas' CSS size and the device pixel ratio.
pub fn sync_window(window: &mut Window) {
    let Some(web) = web_sys::window() else {
        return;
    };
    let Some(canvas) = canvas() else {
        return;
    };
    let (logical_w, logical_h) = (canvas.client_width(), canvas.client_height());
    if logical_w <= 0 || logical_h <= 0 {
        return;
    }
    let dpr = web.device_pixel_ratio().clamp(1.0, 1.5);
    let physical_w = (logical_w as f64 * dpr).round() as u32;
    let physical_h = (logical_h as f64 * dpr).round() as u32;
    if canvas.width() != physical_w {
        canvas.set_width(physical_w);
    }
    if canvas.height() != physical_h {
        canvas.set_height(physical_h);
    }
    if window.resolution.scale_factor() != dpr as f32 {
        window
            .resolution
            .set_scale_factor_override(Some(dpr as f32));
    }
    let (logical_w, logical_h) = (logical_w as f32, logical_h as f32);
    if window.resolution.width() != logical_w || window.resolution.height() != logical_h {
        window.resolution.set(logical_w, logical_h);
    }
}

/// Whether the page holds the pointer lock; the browser drops it on Escape without telling bevy.
pub fn pointer_locked() -> bool {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.pointer_lock_element())
        .is_some()
}

pub fn install_script_hooks() {
    expose_global(
        "spin_command",
        Closure::<dyn Fn(String)>::new(|command: String| {
            COMMANDS.with(|c| c.borrow_mut().push(command));
        }),
    );
    expose_global(
        "spin_status",
        Closure::<dyn Fn() -> String>::new(|| STATUS.with(|s| s.borrow().clone())),
    );
}

/// Commands scripts pushed since the last call, oldest first.
pub fn take_script_commands() -> Vec<String> {
    COMMANDS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

pub fn publish_status(status: &str) {
    STATUS.with(|s| *s.borrow_mut() = status.to_owned());
}

thread_local! {
    static COMMANDS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static STATUS: RefCell<String> = const { RefCell::new(String::new()) };
    static STORAGE: RefCell<Storage> = RefCell::default();
    static WRITES: WriteHooks = WriteHooks {
        committed: Closure::new(|| STORAGE.with(|storage| storage.borrow_mut().committed += 1)),
        failed: Closure::new(|| STORAGE.with(|storage| storage.borrow_mut().failed += 1)),
    };
}

/// The database once all it held has been read back, which is only written to then, so that
/// nothing kept is ever written over unread; what was read until it is handed over; and how the
/// batches written have fared.
#[derive(Default)]
struct Storage {
    database: Option<IdbDatabase>,
    read: Option<Vec<(String, String)>>,
    committed: u32,
    failed: u32,
}

/// What every batch written calls when it has been kept or has failed.
struct WriteHooks {
    committed: Closure<dyn FnMut()>,
    failed: Closure<dyn FnMut()>,
}

fn read_all(request: &IdbOpenDbRequest) {
    let Ok(database) = request
        .result()
        .map(|result| result.unchecked_into::<IdbDatabase>())
    else {
        return opened(Vec::new());
    };
    let reads = database
        .transaction_with_str(STORE)
        .and_then(|transaction| {
            let store = transaction.object_store(STORE)?;
            Ok((transaction, store.get_all_keys()?, store.get_all()?))
        });
    let Ok((transaction, keys, values)) = reads else {
        return opened(Vec::new());
    };
    transaction.set_oncomplete(Some(
        Closure::once_into_js(move || match records(&keys, &values) {
            Some(records) => {
                STORAGE.with(|storage| storage.borrow_mut().database = Some(database));
                opened(records);
            }
            None => opened(Vec::new()),
        })
        .unchecked_ref(),
    ));
    transaction.set_onabort(Some(
        Closure::once_into_js(|| opened(Vec::new())).unchecked_ref(),
    ));
}

/// The records read back, each key with its value, unless the read went wrong.
fn records(keys: &IdbRequest, values: &IdbRequest) -> Option<Vec<(String, String)>> {
    let (keys, values) = (keys.result().ok()?, values.result().ok()?);
    Some(
        js_sys::Array::from(&keys)
            .iter()
            .zip(js_sys::Array::from(&values).iter())
            .filter_map(|(key, value)| Some((key.as_string()?, value.as_string()?)))
            .collect(),
    )
}

fn opened(records: Vec<(String, String)>) {
    STORAGE.with(|storage| storage.borrow_mut().read = Some(records));
}

fn canvas() -> Option<web_sys::HtmlCanvasElement> {
    web_sys::window()?
        .document()?
        .query_selector(CANVAS)
        .ok()??
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .ok()
}

fn expose_global<T: ?Sized + wasm_bindgen::closure::WasmClosure>(name: &str, hook: Closure<T>) {
    if js_sys::Reflect::set(&js_sys::global(), &JsValue::from_str(name), hook.as_ref()).is_err() {
        warn!("could not expose {name} on the page");
    }
    hook.forget(); // hand the closure to JS for the page's lifetime
}
