# Plan

Rewrite this file when a step completes; never append history to it. Keep it short.

## State (`realism`)

Water on the floor is a GPU shallow-water sheet, with drains, flying hoops and jets through portal
mouths, and foam where a jet plunges. The APIC particle solver idles. `game/tests/play_gate.rs`,
the only test, plays 12 fixed scenes and 8 seeded random ones (`scenes_played_at_random`); its
consts must read 3840/2160/10 before any commit or timing. e2e passes on the release web build.
Native runs pipelined (`multi_threaded`), SMAA on the player's view only, per-frame values written
in place (code-map.md "Per-frame values"), a fog on every view (clear above water).

Frame ms p50 at 4K on the RTX 3090, full gate: mound 4.1, random_5 4.7, random_6 5.2, random_7
5.9, big ring 7.4, random_1 8.0, land 8.1, random_8 9.2, dials 9.6, random_2 9.6, wade 9.7,
flood from outside 10.1, trickle 10.2, random_3 10.5, stream 10.8, flood 11.0, random_4 12.1,
waterfall 12.6, dry portals 12.9, portals 23.6. The empty ring reads 3.2–3.6. The CPU issuing a
frame is 3.5–6.2 ms: the scenes are GPU-bound.

## Next

1. random_2 runs 30–44 ms a frame from 18 s on (p95 42): find what it is doing then.
2. GPU per pass, per scene, with the labelled Nsight trace (tools.md): the water's transmissive
   pass, the portal eyes, the transmission snapshots' full-size copies, the shadow cascades.
   Then cut the largest, scene by scene, toward 8.3 ms. DLSS is accepted (tools.md "DLSS").
3. The simulation is not reproducible run to run (lessons.md): what the GPU reads back reaches
   the simulation on no fixed frame. Make it so, so that a seed is a scene.
4. Web: a harness in headed Chrome driven by key and mouse events, measuring frame gaps through
   the page's `?probe` hooks; then 60 fps there.
5. The first-play hitch: a pipeline compiled cold mid-scene stalls a frame for up to seconds.
6. Physics scenes played through the player's inputs: a lake coming level, a fall deflected by
   the spin, momentum through a portal, friction, conservation over long random runs.
7. Seen, not yet worked on: hard-edged screen-space-reflection cut-offs on sheet water; ragged,
   comb-like shorelines; flat dark-teal total-internal-reflection patches seen from under water;
   the dull speckled look of partial or shaded foam; the jet invisible against black space; the
   far rim of a pit zigzagging at quad scale; blocky cap-side skirts by a pit; terraced colour
   bands round a pit; the stair-stepped self-shadow on a pit's sunward slope; the flooded ring
   seen from outside milky with a fine hatch; the sheet and jets do not mirror the viewer's
   figure (only water in flight does).

## Unexplained, watch for

- Once, not reproduced in 3 reruns: 44054 L lying after a 30000 L native pour.
- VRAM creeping ~10 MB/s in the wet portal scene; smeared portal rims; no ground texture in the
  big ring.
