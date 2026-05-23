# CLAUDE.md — World Cup 2026 Predictor

@PROJECT.md

## How to work in this repo

PROJECT.md (imported above) is the single source of truth: architecture (§3–4),
scope (§2), the session-by-session plan (§5), known risks and decisions (§6),
and the session log (§7). **Read it fully before doing anything.** When a design
choice seems ambiguous, check §6 — most non-obvious decisions are recorded there
with their rationale.

## Working rhythm (adapts PROJECT.md §9 for Claude Code)

- Use **plan mode** for any non-trivial change: read the relevant files, propose
  a plan, and wait for approval before editing. This replaces the chat-era rule
  of "one step per response."
- One logical change per commit. Commit messages are scoped and session-prefixed,
  e.g. `Session 26: add "what changed today" diff panel`.
- Show diffs for approval; do not auto-accept edits.
- At session end, propose (a) a one-line session-log entry for §7 and (b) the
  commit message.

## Hard conventions (easy to get wrong)

- Run every script from the **project root**: `python src/<script>.py`,
  `python generate_site.py`. Never from inside `src/`.
- `generate_site.py` is a **pure data-consumer**: it reads CSV/JSON artifacts and
  never imports the model (`src/`) layer. Keep it decoupled.
- `docs/` is **100% build output and must be committed** — GitHub Pages serves
  the site from it.
- Internal team names (Turkey, Czech Republic, Ivory Coast, Cape Verde) belong to
  the **model**; official display names (Türkiye, Czechia, Côte d'Ivoire, Cabo
  Verde) live **only** in the site layer's `DISPLAY_NAMES`. Do not rename in the model.
- `neutral=True` everywhere for v1 (host advantage deferred — see §6).
- Do not hand-edit anything under `data/processed/` — those are script outputs.
  Claude previews/divergences are cached by input hash; only regenerate when
  inputs actually change.
- Python 3.11+, dependencies in `requirements.txt`.

## What this file is

A thin bridge. The detailed plan and history live in PROJECT.md; keep PROJECT.md
current at the end of every session, since it's also the handoff between Claude
Code and the planning chat.