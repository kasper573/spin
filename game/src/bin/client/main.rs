mod platform;

// wasm-bindgen runs main during the page's `init()`; on native targets the browser imports panic.
fn main() {
    console_error_panic_hook::set_once();
    game::systems::app::build(platform::WebPlatform::install(), platform::primary_window()).run();
}
