# Code map

## Frames

The frame's origin is the site on the floor; the ring's axis runs through `[-radius, 0, 0]` along
y. Scripted commands (Pour, Portal, Sculpt) take an absolute phi/y.

## The sheet (`core/sheet/`, glue in `systems/drum/sheet.rs`)

Kurganov–Petrova (2007) central-upwind shallow water in the ring's frame, on the landscape's
cells: a 1024x256 window, closed/periodic when the ring fits. On a wider ring each sheet cell takes
2^k landscape cells (`SheetWindow.taken`), so it spans cap to cap; around a long ring it still ends
~512 cells from the water's site (nested levels are future work).

- State = (water per GLASS area, run round, run along): exact volume on a curved ring. Pushes come
  from the ring's potential with the water's own turn: weight uses (spin + u/r)²r (the Eötvös term,
  the only Coriolis term a thin floor sheet has; nothing is missing). Reflecting wall flux, a
  draining limiter, Manning friction with the EARTH's g (friction tied to weight left weightless
  retrograde water piled up for good) plus laminar drag, RK2.
- The GPU paces itself (`quickest` → `pace`, indirect args in a buffer of their own); the CPU
  sends only as many steps as the last reported wave speed asks for (`SheetFrame.steps`).
- Pour: straight down at 3 m/s over a round bell footprint, at rest over the floor.
- Surface: `raise`/`face` write the water shader's own vertex format (`normal.w` = depth), drawn
  by `LyingWater` materials (`units.z = 1`), hidden and unsent while the sheet is empty; terrain
  reads the corners (`lying_over`).
- CPU side: `Sheet::held()` (row sums by readback, which is the gate's volume), `Sheet::watched()`
  (16x16-cell patch about the avatar + wave speed) → `SheetWindow::bearing` = Archimedes + grip as
  a `WaterCoupling`, merged in `Simulation::advance`. `SheetWindow::sunk` decides the eye is under
  water only past `PUPIL` (2 cm).
- `SheetWindow::lay` is the ONE place the sheet is laid (resize, dry pour, `feed_sheet`). On
  resize, `Landscape::resize` returns a `GroundCarry` inside `DrumResized`; kernels `carry_round`
  then `carry_along` move the water with the ground, exactly.
- `shaders/jets.wgsl`: drains (up to 2, Torricelli outflow against the far drain's head) feed
  `hoops` (a ring buffer of rim points) at the twin mouth; `fly` moves them exactly (to the
  inertial frame, a straight line, back); `land` spreads them into the sheet once the water has
  closed over the whole rim. Jets draw with `units.z = 2`, only while `Sheet::flying() > 0`.
  Foam: sheet `state.w` = entrained air per glass area, fed by `land` (Bin's plunging-jet law),
  advected with the fluxes, rising out at BUBBLE_RISE·BUBBLES_SHARE.
- `persistence` keeps the sheet; `persistence::apply` empties it first.

## The particle solver (`core/fluid/`, idle)

APIC particles on a hashed sparse MAC grid (cell = 2 spacings), 24 red/black SOR sweeps + 8
density-settle sweeps, analytic vessel walls, Akinci body samples. One global particle size
(`Resolution.spacing`), coarsening by 2^(1/3) when `MAX_PARTICLES` = 65536 fill. It is kept as the
3D layer to return to; no particles are made now.

## Landscape and its GPU glue (`systems/drum/`)

- `landscape.rs`: 0.25 m cells, sparse 32x32-cell (8 m) patches over a `base` height. `past` holds
  how far past the glass each cell was dug (`Landscape::reach` is signed), so pits end in a clean
  circle; `terrain_mesh` (render.rs) clips boundary quads on the CPU.
- GPU: R32Float patch atlas + Rgba32Sint patch table (`gpu.rs` `GroundTextures`; vessel bind group
  1: 0 `DrumUniform` dynamic, 1 atlas, 2 table), read by `ground_under(p)` in
  `shaders/drum.wgsl`. Edits upload only changed patches.
- Compute wiring (mirror `core/fluid/gpu.rs`): pipelines at RenderStartup, bind groups in
  PrepareBindGroups, dispatch `.in_set(RenderGraphSystems::Render).before(camera_driver)` (outside
  `Render` its GPU time spans are dropped). Buffers are `Handle<ShaderBuffer>` shared with
  materials; per-frame data crosses as an `ExtractResource`.
- `render.rs rebuild` respawns the glass/structure only when (ring, site, standoff) changes; seen
  from outside the site steps as the wheel turns, hence a respawn every ~11 frames there (by
  design).
- The avatar holds upright by the way up where it stands (`upward` in `core/avatar.rs`), not by
  the terrain's normal; its legs are Coulomb friction (slip/dt within grip).

## Portals (`systems/portal/`)

`VANTAGES = 3` (scene.rs). Each mouth = an eye camera (4x MSAA HDR, own lights, shadow maps,
transmission copies) into `Pictures.fresh[i]`, swapped with `stale` every frame (`eyes.rs
read_from`, which also forces every water material to rebind textures each frame), plus a
water-column camera (`systems/water_column.rs`). Eyes are windowed with `Camera::viewport` +
`sub_camera_view` (`window_on`). `SEEN_THROUGH_LAYERS = 64` on the player camera and the eyes: Bevy
splits depth-sorted see-through items by count into that many snapshots, and the water meshes' sort
centres are meaningless (vertex numbers in position.x), so fewer steps let the cap glass overdraw
the lying water.

## Rules

- `core/` never names `systems/`: the one rule `bin/lint.rs` enforces. The `Vessel` trait
  (`core/vessel.rs`) plus the vessel's WGSL module are the abstraction.
- Persistence: `Snapshot` in `systems/persistence.rs`; new state = a `#[serde(default)]` field,
  written in `kept`, read in `apply`. `save_soon` waits on a save in flight.
