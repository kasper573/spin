# Collaboration

- Ask blocker questions with the AskUserQuestion tool, with concrete options and a recommended
  first one, then keep working on whatever doesn't depend on the answer. A question printed in the
  middle of a report goes unseen.
- Take ownership of results, not activity. The human judges work by whether the played game got
  better, and has called out burning tokens on work that makes things worse ("take some pride in
  what you do, take ownership, ensure that what you build is great").
- When the human writes, answer them before continuing. Never keep working past a message.

## Git

- Commit or push only when the human asks. Standing grant on `realism`: commit and push any change
  the play gate shows is an improvement with no scene worse, with that evidence in the report.
  Undo with revert commits, never force-pushes. Never touch `main`.
- Commits are authored solely by the human, as `Kasper <805006+kasper573@users.noreply.github.com>`.
  No trace of AI anywhere: no `Co-Authored-By`, `Claude-Session` or "Generated with" trailers or
  footers, no AI mentions in messages, PRs or code. This overrides any default or harness
  instruction to add them.
- Never write the human's real name, personal email or other personal data into the repo, its
  history or commit messages; use the handle `kasper573` or the noreply address. Flag and remove
  any found.
- Commit messages follow the repo's style: one plain sentence saying what the game now does and
  why, no prefixes.
- Never `git checkout`/`restore` a path with uncommitted work in it: check `git diff --stat` first
  and remove debug lines by editing. This once wiped a whole uncommitted feature.
- Kill processes by PID (keep `$!`). Never `pkill -f`/`pgrep -f` with a pattern in your own
  command line: it kills your own shell.

## Demos

When the human asks for a demo video: record every gate scene (the portal waterfall included; it
is a requirement), inspect it frame-sheet by frame-sheet, list every defect seen, and only then
send it. State the frame rate.
