# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a solid glass drum spinning in zero g, built in Rust on Bevy. Water is a
position-based fluid, rafts are rigid wooden boards with Coulomb friction against the moving glass,
and the water is drawn as an isosurface with a cel-shaded material and foam.

## Commands

| Command      | What it does                                                   |
| ------------ | -------------------------------------------------------------- |
| `just lint`  | `cargo fmt --check`, the layering lint, clippy (native + wasm) |
| `just test`  | the contract tests in `game/tests/`                            |
| `just bench` | a fixed fluid workload, timed per substep                      |
| `just wasm`  | the browser client through `wasm-bindgen` into `target/wasm/`  |
| `just dist`  | the page plus the wasm bundle in `dist/`                       |
| `just serve` | build `dist/` and serve it on http://localhost:8000            |
| `just dev`   | the same with a fast plain-release build, for local iteration  |
| `just e2e`   | build `dist/` and drive it in headless Chrome                  |

Pushes to `main` lint, test, build, run the e2e and deploy `dist/` to GitHub Pages.

## Layout

One crate, `game/`, split into two layers plus thin binaries:

- `src/core/` — reusable primitives that know nothing about the drum: `fluid/` (position-based
  fluid solver and its spatial grid), `rigid/` (rigid boxes and their contacts), `vessel.rs` (the
  container trait both solvers see their walls through), `surface.rs` (isosurface extraction),
  `fly_camera.rs`, `units.rs` (newtypes), `codec.rs` (float arrays in JSON), `math.rs`, and
  `web.rs` (the browser page: canvas, localStorage, pointer lock, script hooks).
- `src/systems/` — the simulation itself: `drum/` (geometry, sculptable landscape, glass and
  terrain rendering), `sim.rs` (the stepped world and its frame budget), `water.rs` and `rafts.rs`
  (rendering), `aim.rs` (crosshair ray), `controls.rs` (mouse and keys), `settings.rs` (dials and
  toggles), `hud.rs` (text overlay), `persistence.rs` (snapshots), `testing.rs` (script commands and
  status), `scene.rs` (camera, lights, stars), `shaders/` (WGSL, embedded in the binary), and
  `app.rs` (plugin assembly).
- `src/bin/` — `client` (the browser app), `bench`, `lint` (core may not reference systems).
- `tests/` — contract tests against the public API.
- `static/` — the page that loads the wasm bundle. `e2e/` — the headless Chrome smoke test.

## Controls

Click the view to take the mouse, then fly like a spacecraft: WASD moves, Space and Shift move up
and down, Q/E rolls and the mouse steers. Escape releases the mouse.

The crosshair aims at the drum's inner surface: the left button injects water there, the right
button places a raft lying flat on it, and the middle button raises the landscape (hold Control to
lower it). New water and rafts start out moving with the glass.

Every setting is a key, listed on screen with its current value. Hold F1–F7 (spin, flow,
viscosity, wall friction, raft friction, brush size, brush rate) and turn the mouse wheel to change
it; G toggles air drag. Backspace chorded with 1, 2 or 3 removes all water, removes all rafts, or
flattens the landscape.

Settings, water, rafts, landscape, spin and camera are saved to localStorage every couple of
seconds and restored on reload.
