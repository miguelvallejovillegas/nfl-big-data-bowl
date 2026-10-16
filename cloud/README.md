# Running the writeup loop on Claude Code cloud

The loop: each run authors N candidate writeups (one per analytical angle), four judge agents score each against the competition rubric (Football 30 / Data Science 30 / Writeup 20 / Viz 20), the best is revised with the judges' feedback and re-judged, and the round is recorded in `state/` and committed. A routine repeats it on a schedule until the weighted score clears a threshold.

Pieces in this repo:
- `CLAUDE.md` — rules and data facts every agent gets.
- `.claude/agents/judge-*.md` — the four judges (Sonnet); `writeup-author.md` — the author/reviser (Opus).
- `.claude/workflows/iterate-writeup.js` — one round. Invoke locally with `Workflow name: iterate-writeup`, args `{"round": 1, "candidates": 3}`.
- `cloud/setup.sh` — environment setup: `uv sync`, data snapshot, optional Kaggle download.
- `cloud/ROUTINE_PROMPT.md` — the prompt to paste into the routine.
- `writeup/check.py` — word/figure/number validator the judges and authors run.

## One-time setup (≈10 minutes)

1. **Push this repo** and connect GitHub to Claude Code (claude.ai/code → settings, or `/web-setup` in the CLI).
2. **Create a cloud environment** at claude.ai/code → environment selector → *Edit cloud environment*:
   - Setup script: `bash cloud/setup.sh`
   - Network access: **Custom**, add `houjxgqtzxfgccxspsje.supabase.co` (data snapshot). Add `www.kaggle.com` and `kaggle.com` only if you want the full game tracking (set `KAGGLE_API_TOKEN` as an environment variable; the 2.3 GB download takes a few minutes).
   - Environment variables: none required for the snapshot.
3. **Try one round interactively first**: start a cloud session on the repo and paste `cloud/ROUTINE_PROMPT.md`. Check the PR it opens and the token usage of that session before scheduling anything.
4. **Create the routine** at claude.ai/code/routines (or `/schedule` in the CLI): repo = this one, environment = the one above, prompt = `cloud/ROUTINE_PROMPT.md`, schedule = every 2–3 hours while credits last (minimum is hourly). Each run is a fresh clone: state is carried in `state/` through the commits it pushes.
5. **Review the PRs.** Each run opens or updates one. Merge the ones that improve the score; the next run starts from the default branch, so merge before the next scheduled run or point the routine at the `claude/` branch.

## Cost control
- One round ≈ 3 authors (Opus) + 12 judges (Sonnet) + 1 reviser + 4 re-judges + 2 small agents ≈ 22 agents. Expect roughly 1–2 M tokens per round; check the first run and adjust `candidates` (2 is fine once angles are settled).
- The cloud-session credit is applied automatically; after it is spent, runs bill against plan usage, so set the routine to end on a date.
- Lower cost knobs: `candidates: 2`, pass `angles` with one angle to only revise, or run the workflow locally.

## Local use
```
uv run python writeup/check.py            # validate the current writeup
# in Claude Code: run workflow "iterate-writeup" with args {"round": 1, "candidates": 2}
```
