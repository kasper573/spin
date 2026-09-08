# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a solid glass drum spinning in zero g, built in Rust on Bevy. Water is a
position-based fluid, rafts are rigid wooden boards with Coulomb friction against the moving glass,
and the water is drawn as an isosurface with a cel-shaded material and foam.

## Commands

| Command      | What it does                                                  |
| ------------ | ------------------------------------------------------------- |
| `just run`   | native window for development                                 |
| `just lint`  | `cargo fmt --check`, the layering lint, clippy (native + wasm) |
| `just test`  | the contract tests in `game/tests/`                           |
| `just bench` | a fixed fluid workload, timed per substep                     |
| `just wasm`  | the browser client through `wasm-bindgen` into `target/wasm/` |
| `just dist`  | the page plus the wasm bundle in `dist/`                      |
| `just serve` | serve `dist/` on http://localhost:8000                        |
| `just e2e`   | drive `dist/` in headless Chrome                              |

Pushes to `main` lint, test, build, run the e2e and deploy `dist/` to GitHub Pages.

## Layout

- `game/src/core/` — reusable primitives: the fluid solver and rigid boxes confined by a `Vessel`,
  the fly camera, isosurface extraction, unit newtypes, the platform adapter trait.
- `game/src/systems/` — the drum (geometry, landscape, rendering), the running simulation, water
  and raft rendering, crosshair aim, controls, the text HUD, persistence, script hooks.
- `game/src/bin/` — `client` (browser, wasm), `desktop` (native window), `bench`, `lint`.
- `game/src/assets/shaders/` — WGSL for the water, glass and star field, embedded in the binary.
- `web/` — the static page that loads the wasm bundle. `e2e/` — the headless Chrome smoke test.

## Controls

Click the view to take the mouse, then fly like a spacecraft: WASD moves, Space and Shift move up
and down, Q/E rolls, the mouse steers and the wheel changes fly speed. Escape releases the mouse.

The crosshair aims at the drum's inner surface: the left button injects water there, the right
button places a raft lying flat on it, and the middle button raises the landscape (hold Control to
lower it).

Every setting is a key, listed on screen with its current value: F1–F7 step the spin, flow,
viscosity, wall friction, raft friction, brush size and brush rate up (hold Control to step down);
T, G and P toggle match-wheel, air drag and pause; Backspace, Delete, L and R remove the water,
remove the rafts, flatten the landscape and reset everything.

Settings, water, rafts, landscape, spin and camera are saved to localStorage every couple of
seconds and restored on reload.
