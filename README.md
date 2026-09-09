# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a ring world: a glass drum spinning in zero g with ground all the way
round its inside, built in Rust on Bevy. You stand on that ground and weigh exactly what you would
on Earth, because the drum spins at precisely the rate that carries you round at one g; eight
thrusters walk you over the ground, lift you off it and roll you, and a hop is a straight line
through space that the curving floor comes back up to meet. Water is a position-based fluid solved in compute
shaders, rafts are rigid wooden boards with Coulomb friction against the moving ground, and the
water is drawn as an isosurface the GPU extracts every frame, with a cel-shaded material and foam.
It needs a browser with WebGPU (current Chrome, Edge, Safari or Firefox); Chrome on Linux only
offers a GPU adapter with `chrome://flags/#enable-vulkan` turned on.

## The ring

| | |
| --- | --- |
| glass radius | 10.5 m |
| width | 12 m |
| ground | 0.5 m deep all round, so the floor is 10 m from the axis |
| spin | 1.0055 rad/s, one turn every 6.25 s, so that your centre of mass rides at 9.80665 m/s² |
| you | 80 kg, eye 1.7 m above the ground, walk 1.5 m/s, thrusters 17.5 m/s² each (1.78 g) |

The spin is derived, not tuned: ω = √(g / r) for the radius your centre of mass stands at. The
HUD shows your measured weight from the ground's push, which reads 1.00 g standing still, more
walking spinward and less walking against the spin, exactly as a ring this small must.

## Commands

| Command      | What it does                                                   |
| ------------ | -------------------------------------------------------------- |
| `just lint`  | `cargo fmt --check`, the layering lint, clippy (native + wasm) |
| `just test`  | the contract tests in `game/tests/`                            |
| `just bench` | a fixed fluid workload, timed per frame on the GPU             |
| `just record`| an mp4 of each thruster firing, then walking, hopping and flying, into `target/record/` |
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
  bodies and the water kernels see their walls through), `avatar.rs` (the body the viewer rides,
  with its thrusters, legs and head), `audio.rs` (the thrusters' voices), `units.rs` (newtypes),
  `codec.rs` (float arrays in JSON),
  `math.rs`, and
  `web.rs` (the browser page: canvas, localStorage, pointer lock, script hooks).
- `src/systems/` — the simulation itself: `drum/` (geometry, sculptable landscape, glass and
  terrain rendering, and the drum as the water's vessel on the GPU), `sim.rs` (the stepped world),
  `water.rs` and `rafts.rs` (rendering), `player.rs` (the camera on the avatar), `thrusters.rs` (the thruster widget and voices), `aim.rs` (crosshair ray), `controls.rs`
  (mouse and keys), `settings.rs` (dials and
  toggles), `hud.rs` (text overlay), `persistence.rs` (snapshots), `testing.rs` (script commands and
  status), `scene.rs` (camera, lights, stars), `shaders/` (WGSL, embedded in the binary), and
  `app.rs` (plugin assembly).
- `src/bin/` — `client` (the browser app), `bench`, `record` (a video of each thruster firing, then
  the avatar walking, hopping and flying, rendered headless), `lint` (core may not reference systems).
- `tests/` — contract tests against the public API, run headless on whatever GPU (or Vulkan
  software driver) the machine has.
- `static/` — the page that loads the wasm bundle. `e2e/` — the headless Chrome smoke test.

## Controls

You are a body in the simulation like everything else: a ballasted sphere with mass, drag,
friction and buoyancy, whose weighted underside always brings it back upright, with eight
thrusters and legs that push against whatever ground it stands on. Click the view to take the
mouse; the mouse turns your head and Escape releases it. W/S, A/D, Space/Shift and Q/E each fire
a thruster: forward and back, left and right, up and down, roll left and roll right. Thrusters
spool up and down over a third of a second, and the cross in the bottom-left corner shows each
one filling as it fires, the bent arms at its shoulders being the roll pair. Every thruster is
the same jet, heard from where it sits on you (the one pushing you forward roars from behind,
the one pushing you left from your right side) at a quarter loudness as soon as it fires and at
full when it is at full. Opposed thrusters cancel each other out; the widget still shows both
firing.

On the ground the horizontal thrust is your legs' orders, and they walk you at walking speed in
that direction. Each thruster pulls 1.78 g, so holding Space lifts you off the floor at 0.78 g,
and in the air every thruster acts on you directly: fly in bursts, and the ground you left keeps
moving under you until you meet it again.

Enter toggles between solid and ghost. Solid, you are carried round with the ground, weigh what
the spin gives you and cannot leave the drum. As a ghost a flight assist brakes you against the
surrounding air and you drift freely in and out of the drum to look at it from outside.

The crosshair aims at the drum's inner surface: the left button injects water there, the right
button places a raft lying flat on it, and the middle button raises the landscape (hold Control to
lower it). New water and rafts start out moving with the ground.

Every setting is a key, listed on screen with its current value. Hold F1–F7 (spin, flow,
viscosity, wall friction, raft friction, brush size, brush rate) and turn the mouse wheel to change
it; G toggles air drag and Enter solid or ghost. Backspace chorded with 0 resets everything to the
initial state; with 1, 2 or 3 it removes all water, removes all rafts, or flattens the landscape
back to the initial ground.

Settings, water, rafts, landscape, spin and the avatar are saved to localStorage every couple of
seconds and restored on reload.
