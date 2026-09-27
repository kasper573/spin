# Realism

How what the game draws is judged real. The human, on the first whitewater, which the agent had
judged good by comparing it to its own previous render: "Looks ridiculous. Like a junior devs first
toy simulation. The fact that you looked at that and thought it looked good means your judgement
is entirely off."

## Rules

- A render is judged only against real footage of the same thing at the same scale, seen the same
  way. Never against an earlier render: "better than before" is not a verdict.
- The judge is a fresh subagent with none of the work's context: no code, no plan, no intent, no
  earlier renders, told not to read anything but the images. The agent that made the render never
  judges it, and does not argue with the verdict.
- Motion is judged from consecutive frames at the footage's rate as well as from stills.
- Judges are primed to find fault: told one of two real plunges was a render, a judge picked one
  (75% sure) and marked its aerated white "not met" and "texture-like". So a verdict is relative:
  the judge scores both images on the same checklist, and a render passes a reference when two
  judges each score it at least as high as the footage on every item and name no tell in it that
  they do not also name in the footage. Anything short of that fails, and the judges' lists are
  the work list.
- Before visual work, the pairs it touches are judged as they are; after it, again. A change is
  kept only if its verdicts pass (the gate's rules on performance and other scenes still hold).
- A reference's scale is written with it. Footage of a thing far larger or smaller than what the
  game shows is not its reference (the base of a 60 m waterfall is not a 1 m fall from a mouth).

## Procedure

1. Pick the references for what the work touches (below) and the gate scene and moment showing
   the same thing from the same kind of view.
2. Stills: the reference and the render (the gate's 4K frame, downscaled), cropped to the same
   subject, HUD and tool cropped out, in random order as `A` and `B`. Keep which is which apart.
3. Motion: 8 consecutive frames of each at the footage's rate (the render from
   `target/playlive/<scene>.mp4`, every other frame against 30 fps footage), cropped the same, as
   a grid.
4. Two judges per pair, all started together, each told: one of A and B is real footage and one a
   real-time render; say which is the render and how sure; list everything that gives it away,
   most glaring first; then for each checklist item say met, partly or not met in A and in B,
   and where; then which tells each shows.
5. The verdicts go into "Standing" below, replacing earlier ones for the same pair.

References are fetched once per machine into `target/realism/`:
`curl -sL -A "spin-research (kasper573)" -o "target/realism/<name>" "https://commons.wikimedia.org/wiki/Special:FilePath/<name>?width=1600"`
(videos without `?width`, for the 1080p original).

## Tells

A render showing any of these fails, whatever else it gets right:

- a jet meeting a pool at a seam, with no crown or splash hiding it
- white that reads as a texture: fixed to the surface, repeating, evenly noisy, or still while the
  water moves
- soft round puffs: cotton wool, steam or cloud shapes with no inner structure
- an even milky haze, white brighter than its lit surroundings, or white on a shadowed side
- a smooth, flat or faceted ribbon where water falls fast
- polygon edges, quads or grid cells showing

## Plunge: a jet from a mouth into a pool

The gate's fall (`waterfall` from 58 s, `portals`): a 0.8 m mouth in the cap about 1 m above a
pool 1 m deep, falling at about 5 m/s. Its real match is a culvert or fountain outlet.

- `Water_flowing_from_fountain.webm` (CC BY 3.0): a ~1 m square outlet throwing a jet ~2 m into a
  shallow pool, 1080p at 30 fps. The main reference: still at 12 s, crop (300,150)-(1800,1080);
  motion from 12 s, crop (560,480)-(1160,1080).
- `Discharge_pipe.jpg` (public domain), `ColemanCreekCulvertOutfall_(51047162086).jpg` (CC BY
  2.0): a pipe outfall into churning water; a sheet over a low drop into a pool.
- `Dundee_Falls_Ohio_video_2016_04_15.ogv` (CC BY-SA 4.0): a ~3 m fall into a pool.
- `Bubbles_in_water.webm` (CC BY-SA 4.0): an entrained bubble plume seen through a tank wall.
- The human's own reference (not on Commons): a clear stream into a pool, seen from the side at
  the waterline: a glassy stream, a churned clear crown where it enters, a plume of distinct
  bubbles with dark rims and bright glints, single drops, and no white foam.

Checklist:

1. Near the mouth the jet is glassy and clear (fed from still water), refracting what is behind it,
   with sharp highlights along it.
2. Falling, its surface breaks into lumps and streaks along the fall and its edges fray into
   ligaments and drops.
3. Where aerated it is white with grey self-shadow between lumps, darker on the side away from the
   light.
4. Where it enters there is no seam: a crown of churned water rises around it, brightest at the jet
   and gone within one or two jet widths, its outline ragged.
5. Distinct drops and splashes fly from the entry, streaked by motion blur.
6. Near the entry the pool is choppy, many small steep wavelets breaking the reflection into
   facets; farther away it is calmer, dark and reflective.
7. Foam on the surface lies in streaks, lace and patches with dark water between, drifting away
   with the flow and thinning.
8. Near the entry the water under the surface is lightened by bubbles, a pale irregular plume seen
   through the surface.
9. Seen from the side under water: distinct bubbles of millimetres to centimetres, dark-rimmed with
   bright glints, dense at the entry and sparse deeper.
10. Moving, the fine structure of the jet and crown is new within one or two frames at 30 fps;
    nothing holds still; foam drifts at the speed of the surface.

## Shallow water over a floor

The gate: `stream`, `flood`, `wade`, `big_ring`.

- `Caustics_Shallow_Water_Pic1.png`, `Caustics_Shallow_Water_Pic2.png`,
  `Caustic_Shallow_Water.webm` (CC BY-SA 4.0): sunlit shallow water over sand.
- `A_fish_and_caustic.jpg` (CC BY-SA 3.0): caustics seen under water.

Checklist:

1. Under direct sun a net of bright thin lines lies on the floor, sharp in shallow water and
   softer deeper, moving with the waves.
2. The floor seen through the water wavers and bends with the surface.
3. Seen steeply the floor shows; at a grazing angle the surroundings are mirrored.
4. Depth tints and darkens the floor smoothly, without bands.
5. Ripples break reflections into many small highlights.

## Water spreading over dry ground

The gate: `trickle`, and the first seconds of `stream` and `flood`.

- `Вода_набегает_на_сухую_поверхность_асфальта.JPG`,
  `Навал_на_границе_лужи_и_сухого_асфальта.JPG` (CC BY-SA 3.0): a sheet of water running over dry
  asphalt; the edge of a puddle.

Checklist:

1. The front is a sharp irregular line, lobed and fingered, not a smooth curve or a soft fade.
2. Its meniscus catches the light as a thin bright or dark rim.
3. Behind it the ground is darker and more saturated than dry ground, and the sheet mirrors the sky
   at grazing angles.

## Gaps

- No gate scene looks at a plunge from the side at the waterline (plunge item 9).

## Standing

Judged 2026-09-27 at 456e966, one judge per pair, scoring the render only (before the relative
rule). The fountain footage, scored as a render in the calibration: plunge items 1, 4, 6, 8 met;
2, 5, 7 partly; 3 not met; tells b, d. Every render below scores lower on nearly every item.

- Plunge, as committed: fails. No crown, the jet a smooth ribbon to the pool, a flat even white
  oval that holds still on the surface, a calm pool, no drops, no plume. Tells a, b, d, e. First
  glance: "a smooth, static, textured white geometric form".
- Plunge, whitewater (clusters of bubbles, foam and drops drawn as soft clouds; not committed,
  deleted): fails. Soft round puffs, an even milky haze, the same smooth jet and calm pool, no
  distinct drops. Tells c, d, e. First glance: "a soft, slow-motion cloud of milky white puffs,
  like steam or cotton candy" (the human: "a junior devs first toy simulation").
- Shallow water (`stream` at 20 s): fails. No caustics; the water an opaque turquoise wash hiding
  the floor; nothing mirrored at grazing angles; few highlights. No tells.
- Spreading (`trickle` at 15 s): fails. The front partly lobed but too smooth, no meniscus rim, the
  ground behind it neither darker nor reflecting; regular parallel streaks in the sheet read as a
  pattern (tell b). The judge also asked for foam, which its reference does not show: dropped.
