use std::cell::RefCell;

use bevy::prelude::*;
use game::core::platform::Platform;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};

const CANVAS: &str = "#glcanvas";

pub struct WebPlatform;

thread_local! {
    static COMMANDS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static STATUS: RefCell<String> = const { RefCell::new(String::new()) };
}

impl WebPlatform {
    /// Exposes `spin_command(json)` and `spin_status()` on the page for scripts.
    pub fn install() -> WebPlatform {
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
        WebPlatform
    }
}

impl Platform for WebPlatform {
    fn load(&self, key: &str) -> Option<String> {
        storage()?.get_item(key).ok()?
    }

    fn save(&self, key: &str, value: &str) {
        if let Some(storage) = storage() {
            let _ = storage.set_item(key, value);
        }
    }

    fn sync_window(&self, window: &mut Window) {
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

    fn pointer_locked(&self, _requested: bool) -> bool {
        web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.pointer_lock_element())
            .is_some()
    }

    fn take_script_commands(&self) -> Vec<String> {
        COMMANDS.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }

    fn publish_status(&self, status: &str) {
        STATUS.with(|s| *s.borrow_mut() = status.to_owned());
    }
}

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        canvas: Some(CANVAS.to_owned()),
        ..default()
    }
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
    js_sys::Reflect::set(&js_sys::global(), &JsValue::from_str(name), hook.as_ref())
        .unwrap_or_else(|_| panic!("expose {name} on the JS global"));
    hook.forget(); // hand the closure to JS for the page's lifetime
}
