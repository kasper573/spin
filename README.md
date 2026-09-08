# Spin-gravity wheel

Live: https://kasper573.github.io/spin/

A browser simulation of a solid glass drum spinning in zero g. Water is a position-based fluid,
rafts are rigid wooden boards with Coulomb friction against the moving glass, and the water is drawn
with a screen-space fluid renderer (smoothed particle depth, thickness, cel shading with foam).

## Scripts

| Command             | What it does                   |
| ------------------- | ------------------------------ |
| `pnpm dev`          | start the Vite dev server      |
| `pnpm build`        | typecheck and build to `dist/` |
| `pnpm preview`      | serve the production build     |
| `pnpm typecheck`    | `tsc --noEmit`                 |
| `pnpm lint`         | oxlint                         |
| `pnpm format`       | prettier, write                |
| `pnpm format:check` | prettier, check only           |

## Layout

- `src/physics/` — simulation: constants, spatial hash, fluid store, PBF solver, raft rigid body,
  contacts, and the world step.
- `src/render/` — three.js view: orbit camera, drum cage and glass shell, rafts, cursor markers, and
  the screen-space water pass with its GLSL sources under `shaders/`.
- `src/app/` — settings store, pointer handling and the frame loop that ties physics to rendering.
- `src/ui/` — SolidJS control panel and readouts.

## Controls

Click the view to take pointer lock, then fly like a spacecraft: WASD moves, Space and Shift (or R/F) move up and down,
Q/E rolls, the mouse steers and the wheel changes fly speed. The crosshair at the centre of the screen
aims at the drum's inner surface: the left button injects water there and
the right button places a raft lying flat on it, and the middle button raises the landscape
there (hold Control to lower it), with brush size and rate in the panel. Escape releases the controls. Settings, water, rafts, landscape, spin and camera are saved to
localStorage every couple of seconds and restored on reload; Reset clears the simulation. In dev mode `window.sim` exposes the running simulation for scripting, e.g.
`sim.inject(x, y, z, count)` and `sim.advance(seconds)`.
