// wasm-bindgen runs main during the page's `init()`.
fn main() {
    console_error_panic_hook::set_once();
    game::systems::app::build().run();
}
