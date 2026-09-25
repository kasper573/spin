# Workflow

The gate rules in CLAUDE.md "Verification" hold. This file adds what was learned applying them.

Why they exist: after ~15 hours of autonomous work the human played the game and found it ruined
(lag, water in spikes, water turning white, water invisible from some angles, a demo waterfall with
no waterfall). Progress had been measured by headless contracts and bench scenes the agent wrote
itself, "found it" was said again and again, fixes were stacked on fixes, and demo videos were sent
unwatched. Green tests meant nothing about the game the human plays.

- Never say "found it", "fixed" or "root cause" without proof: the symptom reproduced, then shown
  gone in the same reproduction, frames and numbers before and after side by side. Until then it is
  a hypothesis, and is called one.
- Three failed hypotheses on one symptom: stop, revert to the last good state, report what is
  known and what is not. No fourth guess.
- Never send the human anything not inspected first: videos frame-sheet by frame-sheet,
  screenshots at full size. List every defect seen, including ones not yet fixable.
- No hidden state between steps: no long-lived scratch files, worktrees, stashes or background
  jobs. `plan.md` is rewritten, never appended to, and stays short.
- Report honestly and briefly: what was seen, what is proven, what is not. Bad news first.
- Test the general case. The fixed gate scenes looked fine while the human, varying water level,
  portal positions and spin, found bugs quickly. Vary parameters (seeded random scenes built from
  the player's own inputs: dials, tools, thrusters) and give each bug found a fixed scene before
  its fix. A seeded sweep (`scenes_played_at_random`) was written on the human's machine but is
  not in the repo.
- A session without a GPU (a cloud container) cannot run the gate. Say so, and never present work
  as proven that only the gate can prove.

## Clean measurements

The human: "It is absolutely critical that you ensure performance measurements are not polluted
by hogged resources." A whole day of numbers was once ~2.2x too high. Proven cause: while the
display sleeps (the human away, i.e. during every long autonomous stretch) the NVIDIA driver holds
the GPU at P3 (~780 MHz) instead of P0 (~1970 MHz). `just gate` guards against this. For any
timing outside it (a single scene, nsys): `just idle`, then `xset -display :1 dpms force on`
(repeat every 20 s on long runs), check `nvidia-smi --query-gpu=pstate --format=csv` reads P0
under load, and that the scene's `empty ring` reading matches the baseline's (~5 ms at 4K on the
human's RTX 3090; ~10 means throttled). One GPU job at a time. Performance is measured at 4K, not
lower. Compare like with like: a scene run alone reads lower than inside a full gate run
(big ring ~10.5 alone vs ~11.4 in the gate). If the human is using the machine (Chrome, Slack,
GPU at P5), time nothing until it is idle again.

## Short iteration loops

The human called the constant waiting on builds, verifications and recordings "very time
inefficient", and a night of 159 commands for one task "insane".

- Build only in `target` (no second `CARGO_TARGET_DIR`: every change compiled twice).
- Local e2e uses the release web build; the fat-LTO `wasm` profile is for deploys only.
- While iterating: small targeted checks, single gate scenes, short browser probes. The full
  `just gate` / `just verify` runs once, before a commit.
- Spend waiting time reading and planning the next defect, not watching logs.
- Don't tunnel-vision on existing code or legacy tests: the human has granted full rewrite
  freedom as long as the gameplay mechanics stay the same. Entrenching a legacy approach to make
  old tests pass is wasted work.
