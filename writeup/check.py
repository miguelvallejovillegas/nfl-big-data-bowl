"""Validate the Kaggle writeup against competition limits and the numbers file.

Usage: uv run python writeup/check.py [writeup/writeup.md]
Exit code 1 on any violation. Prints a JSON summary.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

WORD_LIMIT = 2000
FIGURE_LIMIT = 9  # "fewer than 10 tables or figures"


def main(path: str = "writeup/writeup.md") -> int:
    p = Path(path)
    if not p.exists():
        print(json.dumps({"ok": False, "error": f"{path} not found"}))
        return 1
    text = p.read_text()
    body = re.sub(r"```.*?```", "", text, flags=re.S)  # code blocks do not count
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    words = len(re.findall(r"\b[\w'’-]+\b", body))
    figures = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text))
    tables = len(re.findall(r"^\s*\|.*\|\s*$\n^\s*\|[\s:|-]+\|\s*$", text, flags=re.M))
    sections = re.findall(r"^#{1,3}\s+(.*)$", text, flags=re.M)
    numbers_file = p.parent / "numbers.json"
    missing = []
    if numbers_file.exists():
        known = json.loads(numbers_file.read_text())
        known_vals = {str(v) for v in flatten(known)}
        known_vals |= {v.lstrip("-") for v in known_vals}  # the regex below reads digits without the sign
        for n in set(re.findall(r"(?<![\w.])(\d+\.\d{2,3})(?![\w.])", body)):
            if n not in known_vals and n.lstrip("0") not in known_vals:
                missing.append(n)
    # optional claim-to-key map (claims.json next to the writeup): every numbers.json key must be claimed by a
    # sentence, and its value must appear in the text
    orphans = []
    claims_file = p.parent / "claims.json"
    if numbers_file.exists() and claims_file.exists():
        claims = json.loads(claims_file.read_text())
        plain = re.sub(r"```.*?```", "", text, flags=re.S).replace("−", "-")  # captions count here
        for k, v in json.loads(numbers_file.read_text()).items():
            if k not in claims or str(v).lstrip("-") not in plain:
                orphans.append(k)
    problems = []
    if orphans:
        problems.append(f"{len(orphans)} numbers.json keys not claimed in the text: {sorted(orphans)[:12]}")
    if words > WORD_LIMIT:
        problems.append(f"{words} words > {WORD_LIMIT}")
    if figures + tables > FIGURE_LIMIT:
        problems.append(f"{figures} figures + {tables} tables > {FIGURE_LIMIT}")
    if missing:
        problems.append(f"{len(missing)} decimals not in numbers.json: {sorted(missing)[:12]}")
    if not any("limit" in s.lower() for s in sections):
        problems.append("no Limitations section")
    out = {"ok": not problems, "words": words, "figures": figures, "tables": tables, "sections": sections, "problems": problems}
    print(json.dumps(out, indent=1))
    return 0 if out["ok"] else 1


def flatten(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from flatten(v)
    elif isinstance(o, list):
        for v in o:
            yield from flatten(v)
    else:
        yield o


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
