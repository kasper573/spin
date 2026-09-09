//! The browser page the client runs in: its canvas, storage, pointer lock and the script hooks
//! `spin_command(json)` and `spin_status()` that automation drives the app through.
use std::cell::RefCell;

use bevy::prelude::*;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

const CANVAS: &str = "#glcanvas";

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        canvas: Some(CANVAS.to_owned()),
        ..default()
    }
}

pub fn storage_load(key: &str) -> Option<String> {
    storage()?.get_item(key).ok()?
}

pub fn storage_save(key: &str, value: &str) {
    if let Some(storage) = storage() {
        let _ = storage.set_item(key, value);
    }
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
/// Without a page there is no browser to drop it, and bevy's own grab stands.
pub fn pointer_locked() -> bool {
    let Some(window) = web_sys::window() else {
        return true;
    };
    window
        .document()
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
}

fn canvas() -> Option<web_sys::HtmlCanvasElement> {
    web_sys::window()?
        .document()?
        .query_selector(CANVAS)
        .ok()??
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .ok()
}

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok()?
}

fn expose_global<T: ?Sized + wasm_bindgen::closure::WasmClosure>(name: &str, hook: Closure<T>) {
    if js_sys::Reflect::set(&js_sys::global(), &JsValue::from_str(name), hook.as_ref()).is_err() {
        warn!("could not expose {name} on the page");
    }
    hook.forget(); // hand the closure to JS for the page's lifetime
}
