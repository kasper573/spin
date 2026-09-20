# CLAUDE.md

## General

- Terseness above all: This repo should contain only our own logic. Anything else should be outsourced to well established crates. Ie. we don't want to build an ECS, a graphics engine, a ui framework, etc. We want to build a simulation, and the code in this repo should reflect that.
- Correctness & clarity comes before performance.
- Tests assert on contracts, never implementation details.
- No mitigation fixes or hacks. Refactoring is encouraged: Don't hunt symptoms, fix root causes.
- No paintjobs. Think longterm when adding features. Again, refactoring is encouraged: Don't just layer code on top of code without thinking about the longterm design. Entropy is the enemy.
- Build once, run everywhere. The same binary (applies to all binaries in the repo) should be able to run in any environment. If assets or environment variables are changed the runtime should work anyway. (Note that runtimes may still panic or have degraded behavior if essential assets are missing)
- No hardcoded environment defaults: Panic if an env var is missing or invalid. Makes mistakes loud and obvious and forces environments to be well and explicitly configured. Also aids with the "build once, run everywhere" principle.

## Code style

- Prioritize simplicity, stability (extensible, not brittle), readability — then performance.
- small, simple `macro_rules!` codegen is allowed to reduce boilerplate, but complex macros are entirely forbidden.
- Files read consumer-first: public API at top, private helpers at the bottom.
- No inline tests: every test lives in its crate's `tests/` folder, against the public API.
- Use `Option`/`Result` and sum types over sentinels/casts. No `unsafe` without a justifying comment.
  Avoid `unwrap`/`panic!` off the test path unless an invariant is truly guaranteed.
- Newtype every float/int that carries a precise unit or id (`Seconds`, `Metres`, `RadiansPerSecond`) — never
  semantic type aliases. The reader must not have to guess a unit, and the type replaces a comment.
  Plain primitives are fine only for obvious-to-everyone concepts (e.g. `foam: f32` in [0, 1]) and inside the inner loops of the solvers, where SI units are the stated convention.
- Don't use #[must_use]. Only when clippy recommends it or when it's absolutely critical.
- Use serde and envy for all json/env serialization and deserialization. No custom parsing code. And use the derive macros, not the imperative APIs.
- Aim for single source of truth (however do not conflate this with DRY. Code duplication is allowed and is not the same thing as SSoT).
- Any and all public type names must be intuitive and not ambigious if listed alongside other public types. Do not rely on crate namespacing to disambiguate.
- A folder may never contain only one file, with the exception of common crate root folders like `src`, `static`, `assets`, `templates`, etc.
- Avoid the use of #[cfg]. Only reach for it when it's absolutely necessary, and keep the usage count as low as possible.

## Architecture

The game crate's `src/` is organized into `core/` and `systems/`:

`core/`:

- code that may be reused by all systems
- typically low level systems and primitives (the fluid and rigid body solvers, the shuttle, surface extraction, the browser page glue)
- may not depend on high level systems
- must be abstract and pluggable: systems integrate with core, core never reaches into a system. Never create a `systems::x` that mirrors a `core::x`. If core code seems to need a system, that's a sign core isn't abstract enough — make it extensible (traits, messages, registries, callbacks) and put the game-specific glue in the relevant feature.

`systems/`:

- high level systems and compositions of core primitives
- the majority of our content and mechanics goes here (the drum, water, landscape, controls, hud, persistence)
- may depend on other high level systems

The binaries under `src/bin/` are thin: the client only assembles the app, nothing more.

## Comments

- The default mindset should be: Do not write comments. Write code that is self explanatory.
- The only exception is: You need to explain WHY, not WHAT some code does. However, even then, you should consider refactoring the code so both the WHAT and the WHY becomes obvious. Only use comments as a final excape hatch.
- Never use comments as a way to give feedback to the prompter. This means comments should never refer to prompt specific details. Comments should be timeless and not rely on the reader being the person who prompted you to do some work.
- Don't scatter duplicate comments describing how a specific mechanism works all over the codebase. Keep it in one place, ideally at the implementation of that mechanism. A common source of this type of bad hygiene is re-explaining a mechanism in the workflow, in env files, in call sites, and finally also in the source code implementation of the mechanism.

## Verification

The played game is the only evidence that a change works. `game/tests/play_gate.rs` plays it as a
player does — the default ring, the avatar in its body, tools worked by their keys and buttons,
every frame drawn — and `just gate` runs it and lays each scene's frames out on a sheet in
`target/playgate/<scene>/`.

- Before a change: `just gate`, and keep the sheets and `measured.txt` of every scene.
- After it: `cargo fmt`, `just verify` (no warnings, no failures), then look at every scene's
  sheet beside the one from before, as a critical player would, and compare what was measured.
  A change that makes any scene look or run worse is not done, whatever passes.
- A defect is reproduced in a gate scene before it is worked on, and is fixed when that same scene
  shows it gone. Until then a cause is a hypothesis, and is called one.
- One change at a time, the smallest that tests the hypothesis. A change that does not cure the
  defect is reverted in full before the next is tried: fixes are never stacked on fixes.
- Something new replaces what is there only once it does better than it in every scene of the
  gate; if it does not, it is deleted rather than patched in place.
- Anything a player can do that the gate does not yet play gets a scene before it gets code.
  Tests drive the game through what a player has — keys, buttons, dials — never by putting the
  simulation into a state from outside.
- Measure on an idle machine at full speed only: nothing else building, testing, recording or
  playing while the gate runs, and the display awake (the GPU driver runs the GPU at a fraction
  of its speed while the display sleeps). `just gate` refuses to start on a busy machine, keeps
  the display awake, and fails if the GPU worked below its full performance state. Every scene
  first times the empty default ring (`empty ring … ms` in `measured.txt`): a run in which that
  reading is not what the baseline's is measured the machine rather than the game, and nothing
  of it is kept. Timing runs outside `just gate` are held to the same.
