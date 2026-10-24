"""Render writeup.md from writeup_template.md and the numbers written by analysis.py.

{key} in the template is replaced by all_numbers.json[key]; {{SHORTLIST}} by the 2026 table built from
shortlist_2026.csv. Then numbers.json / claims.json are re-filtered to the numbers quoted in the text (same rule as
analysis.py and writeup/check.py). Called at the end of analysis.py; can be run alone:
    uv run python writeup/candidates/round_15/revised/render.py
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
    rows = ["| Receiver (class, pick) | 2025: team, alignment, games, % vs man, QB | Routes | SOE, raw → shrunk (in) "
            "| P(2026 SOE > 0) | Pick + routes only | Break G, sd |", "|---|---|---|---|---|---|---|"]
    for _, r in sl.iterrows():
        t = tag(r["name"])
        te = ", TE" if r.pos == "TE" else ""
        mark = " §" if r.combine_discounted else ""
        qb = str(r.qb_2025).split(".")[-1] if isinstance(r.qb_2025, str) else "n/a"
        bg = NUMS.get(f"l26_{t}_bg", "n/a")
        rows.append(f"| {r['name']}{te}{mark} ({r.year}, {r.pick}) | {r.team_2025}, {al.get(r.align_2025, 'wide')}, "
                    f"{NUMS[f'l26_{t}_games']}, {NUMS[f'l26_{t}_man']}%, {qb} | {NUMS[f'l26_{t}_routes']} | "
                    f"{NUMS[f'l26_{t}_soe']} → {NUMS[f'l26_{t}_shr']} | {NUMS[f'l26_{t}_pc1']}% | "
                    f"{NUMS[f'l26_{t}_pc0']}% | {bg} |".replace("-", "−"))
    return "\n".join(rows)


txt = TPL.replace("{{SHORTLIST}}", shortlist())
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
