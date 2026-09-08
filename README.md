# Spin-gravity wheel

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

Left drag uses the selected tool (inject, drain or place raft), right drag or Shift orbits the camera,
the wheel zooms. In dev mode `window.sim` exposes the running simulation for scripting, e.g.
`sim.inject(x, y, z, count)` and `sim.advance(seconds)`.
