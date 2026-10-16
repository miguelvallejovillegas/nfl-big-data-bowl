You are running one improvement round of the NFL Big Data Bowl 2027 writeup in this repository. Follow CLAUDE.md.

1. If `data/processed/combine_events.parquet` is missing, run `bash cloud/setup.sh`.
2. Read `state/latest.json`. Let ROUND = (last round + 1), or 1 if there is no state. If `state/latest.json` has `done: true`, stop and report.
3. Run the saved workflow `iterate-writeup` with args `{"round": ROUND, "candidates": 3, "threshold": 8.5}`. It writes the candidates, judges them with the four rubric agents, revises the best, records `state/round_ROUND.json` and commits.
4. If the workflow returned `done: true`, set `"done": true` in `state/latest.json` and commit.
5. Push the branch and open (or update) a pull request titled "Writeup round ROUND — score X.XX" whose body contains: the winner's path, the four judge scores, the judges' top_changes for the next round, and the diff summary of `writeup/writeup.md`.

Do not touch `data/`, do not add Co-Authored-By trailers, do not change the judge agents or the workflow script unless they crash (then fix minimally and say so in the PR).
