# Plan

Rewrite this file when a step completes; never append history to it. Keep it short.

## State (`realism` at 4d9a682)

Water on the floor is a GPU shallow-water sheet, with drains, flying hoops and jets through portal
mouths, and foam where a jet plunges. The APIC particle solver idles. `game/tests/play_gate.rs`,
the only test, has 12 scenes; its consts must read 3840/2160/10 before any commit or timing.
e2e passes on the release web build. Old tests, `bench` and `record` were deleted on the human's
order: don't recreate them.

Baseline (the human's machine, `target/playgate-baseline/`, at 88b01b4 except `portals` and
`trickle`), p50 ms: trickle 13.8, stream 14.8, flood 15.5, outside 13.3, wade 14.0, land 12.2,
dials 12.6, big ring 10.5, dry portals 18.4, wet portals 32.9, waterfall 19.0. 4d9a682 then took
portals from 32.7 to 32.1 and dry portals down 1.3 ms in the same session; the baseline was not
refreshed after it.
Every scene is over the 8.3 ms budget, so `just gate` exits 101 even at baseline. The gate's P0
guard trips on ~2 samples at scene starts every run: compare per-second numbers.

## Next (look and physics first)

1. Terraced colour bands in the dirt round a pit (dials scene, from above).
2. Seen, not yet worked on: hard-edged screen-space-reflection cut-offs on sheet water; ragged,
   comb-like shorelines; flat dark-teal total-internal-reflection patches seen from under water;
   the dull speckled look of partial or shaded foam; the jet invisible against black space (no
   reflection of the lit ring); the far rim of a pit zigzagging at quad scale; blocky cap-side
   skirts by a pit; wade looking down shows blurred stripes and a huge muzzle glow; the
   stair-stepped self-shadow edge on a pit's sunward slope (hypothesis: the 0.25 m cells'
   bilinear heights, cure a bicubic reconstruction; a 4096 shadow map made no difference); the
   flooded ring seen from outside is milky with a fine hatch.
3. Then: the body displacing the sheet (a wake needs displacement, not drag; see lessons.md), a
   re-lay of the bed that only touches changed cells, far terrain chords hiding a thin sheet.
4. Then performance: fewer portal views (the lever), per-frame values in one shared storage buffer
   instead of re-preparing materials, DLSS at a fixed internal resolution, half-res mirror march.

## Unexplained, watch for

- Once, not reproduced in 3 reruns: 44054 L lying after a 30000 L native pour.
- VRAM creeping ~10 MB/s in the wet portal scene; smeared portal rims; no ground texture in the
  big ring.
