# Plan

The ledger of the work: read it first in every session and after every compaction, rewrite it
(never append) whenever an item moves, and lead every report to the human with the state of its
first item. Keep it short.

## Rules of the ledger

- Anything the human reports goes in as item 1 at once, in their words, with what "done" means.
- One item at a time, finished to its "done" before the next is started. What is found on the
  way goes in as a new item below; it is not worked on then.
- "Done" always includes: reproduced in a gate scene, shown gone in it beside the before, no
  scene worse (`just gate`, `just verify`), committed and pushed, and for anything the human can
  see, an inspected live video sent to them and a pass by `realism.md`'s judges.
- An item whose third hypothesis fails is reverted, reported, and set aside with what is known.

## State (`realism`)

Water on the floor is a GPU shallow-water sheet, with drains, flying hoops and jets through portal
mouths. The APIC particle solver idles. `game/tests/play_gate.rs` plays 12 fixed scenes and 8
seeded random ones; `game/tests/over_time.rs` holds the frame-rate and long-lived-world contracts.
Bodies step at a fixed 120 Hz; the world's clock is f64 (`WorldTime`).

Frame ms p50 at 4K on the RTX 3090, full gate at 456e966: mound 4.4, random_5 4.7, random_6 5.0,
random_7 5.7, big ring 7.5, land 8.2, random_1 8.4, wade 8.6, dials 9.3, random_8 9.7, trickle
9.8, random_2 9.9, flood from outside 10.1, random_3 10.4, stream 10.5, flood 10.7, waterfall
11.9, random_4 12.0, dry portals 13.1, portals 21.1. The empty ring reads 3.3–3.6.

## Items

1. **The waterfall's plunge** (human, 2026-09-27): "the stream just clips into the pool, a bit of
   wobble, and an ugly-ass fake-texture-looking foam thing". Their reference: a clear stream into
   a pool, a churned clear mound where it enters, splashes, and a plume of distinct bubbles with
   bright rims. Judged by `realism.md` (plunge): both what is committed and the first attempt
   fail. Hypothesis 1, failed and deleted: clusters of bubbles, foam and drops simulated after
   Ihmsen et al. and drawn as soft clouds under and over the surface ("cotton candy"). What every
   judge named, in both: the jet a smooth ribbon to the pool (no lumps, streaks or frayed edges,
   not glassy at the mouth); no crown and no drops where it enters; a calm pool; white as an even
   static texture or haze. Done when it passes `realism.md` for the plunge, stills and motion.
2. random_2 runs 30–44 ms a frame from 18 s on (p95 38): find what it is doing then.
3. Toward 8.3 ms a frame: GPU per pass per scene with the labelled Nsight trace (tools.md), then
   cut the largest (water's transmissive pass, portal eyes, transmission copies, shadows). DLSS
   is accepted (tools.md "DLSS").
4. Reproducibility: GPU readbacks reach the simulation on no fixed frame, and a scene's water
   diverges from rounding alone (random_4's p95 went 16.2 to 18.6 ms at 456e966 from a changed
   sum of time, its kernels unchanged). Make a seed a scene.
5. Web at 60 fps: a headed-Chrome harness through the page's `?probe` hooks, then the work.
6. The first-play hitch: a pipeline compiled cold mid-scene stalls a frame up to seconds.
7. Physics scenes through the player's inputs: a lake coming level, a fall deflected by the
   spin, momentum through a portal, friction, conservation over long random runs.
8. Seen, not yet worked on: hard-edged screen-space-reflection cut-offs on sheet water; ragged
   shorelines; flat dark-teal total-internal-reflection patches from under water; the jet
   invisible against black space; a pit's far rim zigzagging at quad scale, blocky skirts,
   terraced colour bands and stair-stepped self-shadow; the flooded ring from outside milky with
   a fine hatch; the sheet and jets do not mirror the viewer's figure.
9. Shallow water fails `realism.md`: no caustics, an opaque turquoise wash hiding the floor,
   nothing mirrored at grazing angles, few highlights.
10. Water spreading over dry ground fails `realism.md`: a front too smooth, no meniscus rim, the
    ground behind neither darker nor reflecting, regular parallel streaks in the sheet.

## Unexplained, watch for

- Once, not reproduced in 3 reruns: 44054 L lying after a 30000 L native pour.
- VRAM creeping ~10 MB/s in the wet portal scene; smeared portal rims; no ground texture in the
  big ring.
