# Lessons

## Pitfalls

- Never name a WGSL entry point after a builtin (`cross` killed the web build: naga's WGSL writer
  renames it for WebGPU, and Chrome then can't find the entry point). Reserved words (`from`,
  `set`, …) fail too.
- A buffer bound read-write can't also be an indirect-dispatch source: give indirect args a buffer
  of their own.
- Compute dispatched outside `RenderGraphSystems::Render` loses its GPU time spans.
- Check `just lint`'s own exit status, not through a pipe, before committing.
- Pixel-compare deterministic frames for render changes.
- Web saves: the sheet readback takes ~4 s on the web, so a save must wait on one already in flight.
- A shrinking ring keeps only ground within half its round of the (drifting ghost) eye, by design.
- What an every-frame jump check still flags after the cures is real: the pour starting (muzzle
  glow) and the eye truly crossing the surface.

## Tried and refuted (don't retry without new evidence)

- The pool vanishing from outside: mesh bounds/culling, sun shadow, `NoAutomaticBatching` and sort
  bias were not the cause. The cause was Bevy's transmission steps (see code-map.md).
- Air entrained by the hose's pour (Bin's law): wide pours entrain ~1% and show nothing; the
  trickle showed a dark speckled disc. Reverted. Partial foam draws badly (see plan.md).
- The pit's rim cut in the fragment shader (discard + prepass shader + `AlphaMode::Mask`): same look,
  +0.2–0.8 ms. The CPU clip in `terrain_mesh` replaced it.
- The digging octagon: coarse far sampling alone and slope carry in `terrain_mesh` did not cure it.
  The cure was keeping `past` (how far past the glass each cell was dug).
- A wake from the body dragging the sheet's water (relaxation toward the hull's run): momentum-correct
  but invisible. The body holding water aside as a hull pressure (−g h dP/ds, surface at level + P):
  conserved, but no wake visible in a 12 cm or a 400 m3 pool. A visible wake needs something else.
- The stair-stepped pit shadow: a 4096 shadow map (+1.1 ms, same shape) and sampling sculpted
  ground at cell spacing didn't change its shape.
- A ghost seat as a vantage (it doesn't co-rotate and flies out of the cap); a mound underfoot as a
  stand (it tilted the view, since cured).
- `ClusterConfig::Single` on every camera (kept, 4d9a682): portals p50 32.7 → 32.1, dry portals
  −1.3 ms. Bevy still rebuilds per-view cluster buffers each frame, just small ones.
