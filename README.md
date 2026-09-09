# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a ring world: a glass drum spinning in zero g with ground all the way
round its inside, built in Rust on Bevy. You stand on that ground and weigh exactly what you would
on Earth, because the drum spins at precisely the rate that carries you round at one g; twelve
thrusters walk you over the ground, lift you off it and turn you, and a hop is a straight line
through space that the curving floor comes back up to meet. Water is a position-based fluid solved in compute
shaders, rafts are rigid wooden boards with Coulomb friction against the moving ground, and the
water is drawn as an isosurface the GPU extracts every frame, with a cel-shaded material and foam.
Nothing in the water has a fixed size: the neighbour grid hashes an unbounded space, the
isosurface is extracted only in the blocks the water touches, and past a budget of particles the
water coarsens, keeping its volume with fewer, larger particles, so the ring can be as big and as
full as the dials allow.
It needs a browser with WebGPU (current Chrome, Edge, Safari or Firefox); Chrome on Linux only
offers a GPU adapter with `chrome://flags/#enable-vulkan` turned on.

## The ring

| | |
| --- | --- |
| glass radius | 10.5 m to start (diameter 6–999 m by F6) |
| width | 12 m to start (2–999 m by F7) |
| ground | 0.5 m deep all round, so the floor is 10 m from the axis |
| spin | 1.0055 rad/s, one turn every 6.25 s, so that your centre of mass rides at 9.80665 m/s² |
| you | 80 kg, eye 1.7 m above the ground, walk 1.5 m/s, thrusters 1.78 times the standing gravity each (17.5 m/s² at one g) |

The spin is derived, not tuned: ω = √(g / r) for the radius your centre of mass stands at. The
HUD shows your measured weight from the ground's push, which reads 1.00 g standing still, more
walking spinward and less walking against the spin, exactly as a ring this small must.

## Commands

| Command      | What it does                                                   |
| ------------ | -------------------------------------------------------------- |
| `just lint`  | `cargo fmt --check`, the layering lint, clippy (native + wasm) |
| `just test`  | the contract tests in `game/tests/`                            |
| `just bench` | a fixed fluid workload, timed per frame on the GPU             |
| `just record`| an mp4 of each thruster firing, then walking, hopping, flying, wading and the ring made bigger, into `target/record/` |
| `just wasm`  | the browser client through `wasm-bindgen` into `target/wasm/`  |
| `just dist`  | the page plus the wasm bundle in `dist/`                       |
| `just serve` | build `dist/` and serve it on http://localhost:8000            |
| `just dev`   | the same with a fast plain-release build, for local iteration  |
| `just dev-native` | the client in a native window, at native speed              |
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
- `static/` — the page that loads the wasm bundle; opened with `?probe` it mirrors the app's status into the window title and runs the `cmd=` commands in the URL, for driving a browser that refuses DevTools. `e2e/` — the headless Chrome smoke test.

## Controls

You are a body in the simulation like everything else: a ballasted sphere with mass, drag,
friction and buoyancy, with twelve thrusters and legs that push against whatever ground it
stands on. Your eyes look straight out of the hull, so you turn by turning the hull. Click the
view to take the mouse and Escape releases it. W/S, A/D, Space/Shift and Q/E each fire a
thruster: forward and back, left and right, up and down, roll left and roll right. The mouse
fires the pitch and yaw pairs: moving it asks for turn in that direction, at full when it moves
fast, and the turn stops when the mouse does. Gyros hold whatever attitude the turning thrusters
leave you in: standing, they carry it round with the ground under your feet, so walking keeps you
as upright as you stood; in the air and afloat they carry it with the drum's air, so a hop lands
you tilted by the angle you flew round the ring and a spell adrift in water moving against the
drum leaves you leaning, until you level yourself again with the mouse. Outside the drum nothing
turns you but your own thrusters and whatever you bump into. Thrusters spool up and down
over a third of a second, and the cross in the bottom-left corner shows each pushing one where it
sits on you, filling as it fires: pushing forward lights the arm at the back. Every pushing
thruster is the same jet, heard from where it sits (the one pushing you forward roars from
behind, the one pushing you left is louder in your right ear) at a quarter loudness as soon as it
fires and at full when it is at full. The turning thrusters are silent and not shown.
Opposed thrusters cancel each other out; the widget still shows both firing.

On the ground the horizontal thrust is your legs' orders, and they walk you at walking speed in
that direction. Thruster power is a setting (F8), and starts out equalized to the standing
gravity: each thruster pulls 1.78 times it, so holding Space lifts you off the floor at 0.78 g,
and in the air every thruster acts on you directly: fly in bursts, and the ground you left keeps
moving under you until you meet it again. In water you float, only just, and the water drags on
you: the thrusters still push you through it and out of it, at a few metres a second.

Enter toggles between solid and ghost. Solid, you are carried round with the ground, weigh what
the spin gives you and cannot leave the drum. As a ghost a flight assist brakes you against the
surrounding air and you drift freely in and out of the drum to look at it from outside.

The crosshair aims at the drum's inner surface: the left button injects water there, the right
button places a raft lying flat on it, and the middle button raises the landscape (hold Control to
lower it). New water and rafts start out moving with the ground.

Every setting is a key, listed on screen with its current value. Hold F1–F8 (spin, flow,
viscosity, wall friction, raft friction, ring diameter, ring width, thruster power) and turn the
mouse wheel to change it, in steps that grow with the value so the top of a dial is a few hundred
clicks away; G toggles air drag and Enter solid or ghost. Changing the ring's size
keeps everything else: the water and the landscape stretch to fit, and whatever solid the new
walls would cut through is pulled inside them; a ghost stays where it is. The spin stays, so a bigger ring pulls harder; T
equalizes the thruster power to the standing gravity again, the way the initial state is set up,
so the game plays as it did at the start. Backspace chorded with 0 resets everything to the
initial state; with 1, 2 or 3 it removes all water, removes all rafts, or flattens the landscape
back to the initial ground.

Settings, water, rafts, landscape, spin and the avatar are saved to localStorage every couple of
seconds and restored on reload.
