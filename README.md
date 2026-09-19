# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a ring world: a glass drum spinning in zero g with ground all the way
round its inside, spun at exactly the rate that gives you your Earth weight on that ground. You
are a body in it, flown on thrusters, with a tool floating before you: a number key brings one
out and puts it away again, and with them you can flood it with water, sculpt the landscape and
join two places of it with a pair of portals, which bodies, water and light go through as they
would through any opening, then resize the ring, whose wall keeps the ground and the water where they lie on it. The water is a position-based fluid solved in compute shaders and drawn as an isosurface
the GPU extracts every frame; the amount of water and the size of the ring are only limited by
the dials.

Built in Rust on Bevy, for WebGPU. It runs in a current Chrome, Edge, Safari or Firefox; Chrome
on Linux only offers a GPU adapter with `chrome://flags/#enable-vulkan` turned on. Every control
and setting that is not a tool's is listed on screen, with its current value; a tool shows its
own on the screen in its back, and answers to the mouse buttons and the wheel while it is out.

## Working in the repo

| Command           | What it does                                                        |
| ----------------- | ------------------------------------------------------------------- |
| `just lint`       | `cargo fmt --check`, the layering lint, clippy (native and wasm)     |
| `just test`       | the contract tests in `game/tests/`, headless on the machine's GPU  |
| `just bench`      | a fixed fluid workload, timed per frame on the GPU                  |
| `just wasm`       | the browser client through `wasm-bindgen` into `target/wasm/`       |
| `just dist`       | the page plus the wasm bundle in `dist/`                            |
| `just serve`      | build `dist/` and serve it on http://localhost:8000                 |
| `just dev`        | the same with a fast plain-release build, for local iteration       |
| `just dev-native` | the client in a native window                                       |
| `just e2e`        | build `dist/` and drive it in headless Chrome                       |
| `just record`     | an mp4 of the avatar put through its paces, into `target/record/`   |

Pushes to `main` lint, test, build, run the e2e and deploy `dist/` to GitHub Pages.

## Layout

One crate, `game/`, in two layers plus thin binaries:

- `src/core/` — primitives that know nothing about the ring world: the fluid solver, body
  coupling and isosurface extraction (WGSL compute kernels and what dispatches them), rigid
  bodies and their contacts, the vessel both see their walls through, the avatar, the thrusters'
  voices, units and the browser page glue. Core may not reference `systems/`; `just lint`
  checks.
- `src/systems/` — the simulation itself: the drum and its landscape, the stepped world, water
  rendering, the player's camera and body, the tools and what the mirrors show of them, the
  portals and what is seen and lit through them, controls, settings, HUD, persistence, and the script
  commands and status the tests and the page's `?probe` mode drive it through.
- `src/bin/` — the browser client, the bench, the recorder and the layering lint.
- `tests/` — contract tests against the public API, run headless.
- `static/` — the page that loads the wasm bundle; `e2e/` — the headless Chrome smoke test.

Each module's doc comment explains what it does and why; the code is meant to be read.
