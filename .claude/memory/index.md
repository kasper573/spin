# Memory

Learned by agents, kept by agents (see "Memory" in CLAUDE.md). Each file holds one subject; a
learning lives in exactly one of them. Rewrite in place rather than append history: a file states
what is true now. Never write secrets, credentials or personal data here.

Read at the start of every session:

- @.claude/memory/workflow.md — how work is judged and done: evidence from the played game, one
  change at a time, clean measurements, short iteration loops
- @.claude/memory/collaboration.md — how the human wants to be worked with: questions, commits,
  authorship, ownership
- @.claude/memory/direction.md — the goal and the human's product decisions
- @.claude/memory/plan.md — where the work stands and what comes next

Read before the work they cover:

- [code-map.md](code-map.md) — the sheet, the idle particle solver, the landscape and its GPU
  glue, portals, persistence, frames
- [tools.md](tools.md) — running single gate scenes, profiling with Nsight, the parked DLSS state,
  e2e in headless Chrome
- [lessons.md](lessons.md) — technical pitfalls, and approaches tried and refuted (so they are not
  tried again blind)
