"""Render writeup.md from writeup_template.md and the numbers written by analysis.py.

{key} in the template is replaced by all_numbers.json[key]; {{SHORTLIST}} by the 2026 table built from
shortlist_2026.csv. Then numbers.json / claims.json are re-filtered to the numbers quoted in the text (same rule as
analysis.py and writeup/check.py). Called at the end of analysis.py; can be run alone:
    uv run python writeup/candidates/round_16/cand_2/render.py
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


def tag(name):
    return name.split()[-1].lower().replace("'", "").replace(".", "")


def shortlist():
    sl = pd.read_csv(OUT / "shortlist_2026.csv")
    sl = sl.sort_values("p_recal", ascending=False)
    al = {"WR": "wide", "SLOT_WR": "slot", "TE": "inline"}
    rows = ["| Receiver (class, pick) | 2025 role: team, alignment, % vs man, QB | Games | Routes | Shrunk SOE, in "
            "| P(2026 SOE > 0) | Break G, sd |", "|---|---|---|---|---|---|---|"]
    for _, r in sl.iterrows():
        t = tag(r["name"])
        te = ", TE" if r.pos == "TE" else ""
        mark = " §" if r.combine_discounted else ""
        act = "**" if r.act else ""
        flag = " ⚑" if r.avail_flag else ""
        qb = str(r.qb_2025).split(".")[-1] if isinstance(r.qb_2025, str) else "n/a"
        bg = NUMS.get(f"l26_{t}_bg", "n/a")
        rows.append(f"| {act}{r['name']}{act}{te}{mark} ({r.year}, {r.pick}) | {r.team_2025}, "
                    f"{al.get(r.align_2025, 'wide')}, {NUMS[f'l26_{t}_man']}%, {qb} | {NUMS[f'l26_{t}_games']}{flag} | "
                    f"{NUMS[f'l26_{t}_routes']} | {NUMS[f'l26_{t}_shr']} | {NUMS[f'l26_{t}_pc1']}% | {bg} |"
                    .replace("-", "−"))
    return "\n".join(rows)


def acton():
    sl = pd.read_csv(OUT / "shortlist_2026.csv").sort_values("p_recal", ascending=False)
    a = sl[sl.act]
    out = []
    for _, r in a.iterrows():
        key = "l26_" + tag(r["name"]) + "_pc1"
        out.append(f"{r['name']} ({r.team_2025}, {NUMS[key]}%)")
    return ", ".join(out)


def history():
    h = pd.read_csv(OUT / "rule_history.csv")
    out = []
    for s1, sub in h.groupby("s1"):
        names = ", ".join(f"{n.split()[-1]} {'✓' if hit else '✗'}" for n, hit in zip(sub.display_name, sub.hit))
        out.append(f"{s1}→{str(s1 + 1)[2:]}: {names}")
    return "; ".join(out)


txt = TPL.replace("{{SHORTLIST}}", shortlist()).replace("{{ACTON}}", acton()).replace("{{HISTORY}}", history())
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
num = {k: v for k, v in NUMS.items() if k in used or (k.startswith("l26_") and quoted(v))}
num = {k: v for k, v in num.items() if quoted(v)}
(OUT / "numbers.json").write_text(json.dumps(num, indent=1))
(OUT / "claims.json").write_text(json.dumps({k: CLAIMS.get(k, k) for k in num}, indent=1))
print(f"render: writeup.md written; {len(num)} quoted numbers")
