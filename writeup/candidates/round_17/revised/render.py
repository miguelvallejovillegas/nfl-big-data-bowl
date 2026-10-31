"""Render writeup.md from writeup_template.md and the numbers written by analysis.py.

{key} in the template is replaced by all_numbers.json[key]. Blocks built from the CSVs written by analysis.py:
{{SHORTLIST}} (act-on names: GO / HOLD table), {{CLIPS}} (film checklist for the GO names, clips_2026.csv),
{{REST}} (the rest of the 2026 list). Every numbers.json key read by a block is recorded, so numbers.json holds exactly
the keys the text uses (one key per quoted number). Called at the end of analysis.py; can be run alone:
    uv run python writeup/candidates/round_17/revised/render.py
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
USED: set[str] = set()


def N(key):
    USED.add(key)
    return NUMS[key]


def tag(name):
    return name.split()[-1].lower().replace("'", "").replace(".", "")


def qb(v):
    return str(v).split(".")[-1] if isinstance(v, str) else "n/a"


def sl_sorted():
    return pd.read_csv(OUT / "shortlist_2026.csv").sort_values("p_recal", ascending=False)


def shortlist():
    sl = sl_sorted()
    rows = ["| Call | Receiver (class, pick; 2025 team, QB) | 2025 games, routes; slot share | Shrunk SOE; P | "
            "Earlier lists |", "|---|---|---|---|---|"]
    for _, r in sl[sl.act].sort_values(["go", "p_recal"], ascending=False).iterrows():
        t = tag(r["name"])
        hist = r["hist"] if isinstance(r["hist"], str) and r["hist"] else "first listing"
        rows.append(f"| **{'GO' if r.go else 'HOLD'}** | **{r['name']}**, {r.pos} ({r.year}, {r.pick}; {r.team_2025}, "
                    f"{qb(r.qb_2025)}) | {N(f'l26_{t}_games')} g, {N(f'l26_{t}_routes')}; {N(f'sl_{t}_slot')}% slot | "
                    f"{N(f'l26_{t}_shr')} in; {N(f'l26_{t}_pc1')}% | {hist} |".replace("-", "−"))
    return "\n".join(rows)


def clips():
    sl = sl_sorted()
    cl = pd.read_csv(OUT / "clips_2026.csv")
    lg = {"hitch": float(NUMS["vap_hitch"]), "slant": float(NUMS["vap_slant"]), "out": float(NUMS["vap_out"])}
    out = []
    for _, r in sl[sl.act & sl.go].iterrows():
        t = tag(r["name"])
        items = []
        for c in cl[cl.tag == t].itertuples():
            when = f"{c.season} wk {c.week} vs {c.opp}, Q{c.q} {c.clock.lstrip('0')}, {c.down} & {c.ytg}, {c.cov}"
            v, sep = N(f"clip_{t}_{c.k}_v"), N(f"clip_{t}_{c.k}_sep")
            if c.what == "sink":
                flag = "" if float(v) <= lg["hitch"] else " **No sink: check his other hitches first.**"
                items.append(f"*Hitch sink* ({when}): {v} yd/s at the top, {sep} yd open.{flag}")
            elif c.what == "speed":
                flag = "" if float(v) >= lg[c.route] else " **Slowed below the league median: check.**"
                items.append(f"*{c.route.capitalize()} at speed* ({when}, {c.align}): {v} yd/s at the turn, "
                             f"{sep} yd open.{flag}")
            else:
                items.append(f"*Against man* ({when}, {c.route}): {sep} yd open.")
        out.append(f"- **{r['name']}.** " + " ".join(f"({i + 1}) {x}" for i, x in enumerate(items)))
    return "\n".join(out)


def rest():
    sl = sl_sorted()
    out = []
    for _, r in sl[~sl.act].iterrows():
        t = tag(r["name"])
        te = ", TE" if r.pos == "TE" else ""
        flag = " ⚑" if r.avail_flag else ""
        out.append(f"{r['name']}{te}{flag} ({r.team_2025}; {N(f'l26_{t}_shr')} in; {N(f'l26_{t}_pc1')}%)"
                   .replace("-", "−"))
    return "; ".join(out) + ". ⚑ = 13 or fewer games in 2025."


txt = TPL.replace("{{SHORTLIST}}", shortlist()).replace("{{CLIPS}}", clips()).replace("{{REST}}", rest())
missing = sorted(set(re.findall(r"\{([a-z0-9_]+)\}", txt)) - set(NUMS))
if missing:
    raise KeyError(f"template keys not in all_numbers.json: {missing}")
USED |= set(re.findall(r"\{([a-z0-9_]+)\}", txt))
txt = re.sub(r"\{([a-z0-9_]+)\}", lambda m: NUMS[m.group(1)].replace("-", "−"), txt)
(OUT / "writeup.md").write_text(txt)

# numbers.json = exactly the keys the text uses (one key per quoted number, nothing else)
num = {k: NUMS[k] for k in sorted(USED)}
(OUT / "numbers.json").write_text(json.dumps(num, indent=1))
(OUT / "claims.json").write_text(json.dumps({k: CLAIMS.get(k, k) for k in num}, indent=1))
print(f"render: writeup.md written; {len(num)} numbers in numbers.json")
