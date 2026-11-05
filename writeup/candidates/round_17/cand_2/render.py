"""Render writeup.md from writeup_template.md and the numbers written by analysis.py.

{key} in the template is replaced by all_numbers.json[key]; {{SHORTLIST}} (the four act-on names with scout notes),
{{WATCH}}, {{REST}} (the rest of the 2026 list), {{ACTON}} and {{HISTORY}} are built from shortlist_2026.csv,
scout_2026.csv and rule_history.csv. Then numbers.json / claims.json are re-filtered to the numbers quoted in the text
(same rule as analysis.py and writeup/check.py). Called at the end of analysis.py; can be run alone:
    uv run python writeup/candidates/round_17/cand_2/render.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
NUMS = json.loads((OUT / "all_numbers.json").read_text())
CLAIMS = json.loads((OUT / "all_claims.json").read_text()) if (OUT / "all_claims.json").exists() else {}
TPL = (OUT / "writeup_template.md").read_text()
AL = {"WR": "wide", "SLOT_WR": "slot", "TE": "inline"}


def tag(name):
    return name.split()[-1].lower().replace("'", "").replace(".", "")


def qb(v):
    return str(v).split(".")[-1] if isinstance(v, str) else "n/a"


def sl_sorted():
    return pd.read_csv(OUT / "shortlist_2026.csv").sort_values("p_recal", ascending=False)


def shortlist():
    sl = sl_sorted()
    sc = pd.read_csv(OUT / "scout_2026.csv").set_index("tag")
    rows = ["| Call | Receiver (class, pick; 2025 team, QB) | 2025 games, routes; shrunk SOE; P | Earlier lists | Film first |",
            "|---|---|---|---|---|"]
    for _, r in sl[sl.act].sort_values(["go", "p_recal"], ascending=False).iterrows():
        t = tag(r["name"])
        s = sc.loc[t]
        hist = r["hist"] if isinstance(r["hist"], str) and r["hist"] else "first listing"
        film = f"{s.r1.capitalize()} {NUMS[f'sc_{t}_r1p']}% of sampled routes; man {NUMS[f'l26_{t}_man']}%"
        pick = r.pick if str(r.pick) != "UDFA" else "UDFA"
        rows.append(f"| **{'GO' if r.go else 'HOLD'}** | **{r['name']}** ({r.year}, {pick}; {r.team_2025}, {qb(r.qb_2025)}) | "
                    f"{NUMS[f'l26_{t}_games']} g, {NUMS[f'l26_{t}_routes']}; {NUMS[f'l26_{t}_shr']} in; "
                    f"{NUMS[f'l26_{t}_pc1']}% | {hist} | {film} |".replace("-", "−"))
    return "\n".join(rows)


def watch():
    sl = sl_sorted()
    out = []
    for _, r in sl[~sl.act].head(4).iterrows():
        t = tag(r["name"])
        flag = f", {NUMS[f'l26_{t}_games']} games" if r.avail_flag else ""
        out.append(f"{r['name']}{', TE' if r.pos == 'TE' else ''} ({NUMS[f'l26_{t}_pc1']}%{flag})")
    return ", ".join(out)


def rest():
    sl = sl_sorted()
    out = []
    for _, r in sl[~sl.act].iterrows():
        t = tag(r["name"])
        te = ", TE" if r.pos == "TE" else ""
        flag = " ⚑" if r.avail_flag else ""
        out.append(f"{r['name']}{te} ({r.year}, {r.pick}; {r.team_2025}; {NUMS[f'l26_{t}_games']} g{flag}; "
                   f"{NUMS[f'l26_{t}_shr']} in; {NUMS[f'l26_{t}_pc1']}%)".replace("-", "−"))
    return "; ".join(out) + ". ⚑ = 13 or fewer games in 2025."


def acton():
    sl = sl_sorted()
    return " and ".join(f"{r['name']} ({r.team_2025})" for _, r in sl[sl.act & sl.go].iterrows())


def history():
    h = pd.read_csv(OUT / "rule_history.csv")
    rows = ["| List (season → next) | Hits | Misses |", "|---|---|---|"]
    for s1, sub in h.groupby("s1"):
        hits = ", ".join(n for n, x in zip(sub.display_name, sub.hit) if x) or "none"
        miss = ", ".join(n for n, x in zip(sub.display_name, sub.hit) if not x) or "none"
        rows.append(f"| {s1} → {str(s1 + 1)[2:]} ({sub.hit.sum()} of {len(sub)}) | {hits} | {miss} |")
    return "\n".join(rows)


txt = (TPL.replace("{{SHORTLIST}}", shortlist()).replace("{{ACTON}}", acton()).replace("{{HISTORY}}", history())
       .replace("{{WATCH}}", watch()).replace("{{REST}}", rest()))
missing = sorted(set(re.findall(r"\{([a-z0-9_]+)\}", txt)) - set(NUMS))
if missing:
    raise KeyError(f"template keys not in all_numbers.json: {missing}")
txt = re.sub(r"\{([a-z0-9_]+)\}", lambda m: NUMS[m.group(1)].replace("-", "−"), txt)
(OUT / "writeup.md").write_text(txt)

plain = txt.replace("−", "-")


def quoted(v):
    s_ = str(v).lstrip("-")
    pat = r"(?<![\d.])" + ("" if s_.startswith("+") else "[+-]?") + re.escape(s_) + r"(?![\d]|\.\d)"
    return re.search(pat, plain) is not None


used = set(re.findall(r"\{([a-z0-9_]+)\}", TPL))
num = {k: v for k, v in NUMS.items() if k in used or (k.startswith(("l26_", "sc_")) and quoted(v))}
num = {k: v for k, v in num.items() if quoted(v)}
(OUT / "numbers.json").write_text(json.dumps(num, indent=1))
(OUT / "claims.json").write_text(json.dumps({k: CLAIMS.get(k, k) for k in num}, indent=1))
print(f"render: writeup.md written; {len(num)} quoted numbers")
