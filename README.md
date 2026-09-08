# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a solid glass drum spinning in zero g, built in Rust on Bevy. Water is a
position-based fluid solved in compute shaders, rafts are rigid wooden boards with Coulomb friction
against the moving glass, and the water is drawn as an isosurface the GPU extracts every frame,
with a cel-shaded material and foam. It needs a browser with WebGPU (current Chrome, Edge, Safari
or Firefox).

## Commands

| Command      | What it does                                                   |
| ------------ | -------------------------------------------------------------- |
| `just lint`  | `cargo fmt --check`, the layering lint, clippy (native + wasm) |
| `just test`  | the contract tests in `game/tests/`                            |
| `just bench` | a fixed fluid workload, timed per frame on the GPU             |
| `just wasm`  | the browser client through `wasm-bindgen` into `target/wasm/`  |
| `just dist`  | the page plus the wasm bundle in `dist/`                       |
| `just serve` | build `dist/` and serve it on http://localhost:8000            |
| `just dev`   | the same with a fast plain-release build, for local iteration  |
| `just e2e`   | build `dist/` and drive it in headless Chrome                  |

Pushes to `main` lint, test, build, run the e2e and deploy `dist/` to GitHub Pages.

## Layout

One crate, `game/`, split into two layers plus thin binaries:

- `src/core/` — reusable primitives that know nothing about the drum: `fluid/` (the position-based
  fluid solver, body coupling and isosurface extraction as WGSL compute kernels under `shaders/`,
  with the buffers, pipelines and per-frame dispatch that run them), `rigid/` (boxes and ballasted
  spheres, their contacts, and what the water does to them), `vessel.rs` (the container both the
  bodies and the water kernels see their walls through), `shuttle.rs` (the self-righting vehicle
  the viewer rides), `units.rs` (newtypes), `codec.rs` (float arrays in JSON), `math.rs`, and
  `web.rs` (the browser page: canvas, localStorage, pointer lock, script hooks).
- `src/systems/` — the simulation itself: `drum/` (geometry, sculptable landscape, glass and
  terrain rendering, and the drum as the water's vessel on the GPU), `sim.rs` (the stepped world),
  `water.rs` and `rafts.rs` (rendering), `player.rs` (the camera on the shuttle), `aim.rs` (crosshair ray), `controls.rs`
  (mouse and keys), `settings.rs` (dials and
  toggles), `hud.rs` (text overlay), `persistence.rs` (snapshots), `testing.rs` (script commands and
  status), `scene.rs` (camera, lights, stars), `shaders/` (WGSL, embedded in the binary), and
  `app.rs` (plugin assembly).
- `src/bin/` — `client` (the browser app), `bench`, `lint` (core may not reference systems).
- `tests/` — contract tests against the public API, run headless on whatever GPU (or Vulkan
  software driver) the machine has.
- `static/` — the page that loads the wasm bundle. `e2e/` — the headless Chrome smoke test.

## Controls

You ride a small shuttle that the simulation treats like everything else: a ballasted sphere with
mass, drag, friction and buoyancy, whose weighted underside always brings it back upright. Click
the view to take the mouse, then fly: WASD thrusts, Space and Shift thrust up and down, Q/E rolls
and the mouse turns your head. Let go of the keys and the flight assist brakes you against the
surrounding air. Escape releases the mouse.

Enter toggles the shuttle's collisions. Off, it is a ghost that everything except walls acts on,
free to drift in and out of the drum. On, it is solid: outside it lands on the glass, inside it is
carried round with the air, falls to the floor under the spin and cannot leave.

The crosshair aims at the drum's inner surface: the left button injects water there, the right
button places a raft lying flat on it, and the middle button raises the landscape (hold Control to
lower it). New water and rafts start out moving with the glass.

Every setting is a key, listed on screen with its current value. Hold F1–F7 (spin, flow,
viscosity, wall friction, raft friction, brush size, brush rate) and turn the mouse wheel to change
it; G toggles air drag and Enter the shuttle's collisions. Backspace chorded with 1, 2 or 3 removes all water, removes all rafts, or
flattens the landscape.

Settings, water, rafts, landscape, spin and the shuttle are saved to localStorage every couple of
seconds and restored on reload.
