# Direction

## The goal

A realtime simulation of a spin-gravity ring world with editable water, land and portals, at
perfect realism: looks and behaviour near indistinguishable from reality, every system built from
first principles so that emergent behaviour matches reality (e.g. air bubbles under water from
waves, from the body, from a stream a portal drives down). No heuristics standing in for physics;
"realistic enough" does not satisfy it. It must hold at any scale: every editable parameter can take
any value. The only concession to reality is portals: they add no gravity and need no black holes,
and otherwise act on gravity, mass, light, air and water exactly as physics predicts.

## Priorities (the human's decisions, newest first)

- Look and physics first; speed stays parked until the game looks and behaves right. Visual
  glitches (white circles, large globules, spikes) outrank frame time.
- The human was happy with the realism level of the 2026-09-21 demo, apart from specific defects
  (since cured: water cutting to a new state, pool vanishing from outside, missing waterfall).
- Performance targets: 120–240 fps native at 4K; a stable 60 fps in Chrome. Reason about which
  designs can reach that with unbounded scale rather than blindly optimizing toward the number.
  DLSS is accepted.
- Use established tools and practices (Rust, Bevy, naga, profilers) and look them up, rather than
  inventing methods.

## Product decisions

- Tools act at the destination. The water tool throws no ballistic jet: water appears at the aimed
  spot, and so do the effects of the land and portal tools. Tools are the player's hands, not
  simulated devices. Never propose projectile behaviour for tools.
- The water bed is blue-lagoon silt, so light through the water gives its blue look (a real
  phenomenon, keep it).
- The portal waterfall is a requirement and the most complex proof of visual and behavioural
  correctness: a 1 m pool with a mouth in its bed, its twin on the same axis in the wall/cap 1 m
  above the waterline; spin gravity drives an endless fall, ideally a laminar stream.
- The white ellipse on the bed under a portal pair is the sunbeam through the portals: intended.
- Hard-edged square pours, black holes in the ground and cube-like water bodies look "absurd": never
  ship them.
- Digging must end in a clean circle, not the tiling of the cells.
