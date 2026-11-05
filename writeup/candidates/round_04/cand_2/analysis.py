"""Speed Kept (SK): WR/TE break mechanics, Combine route drills vs the same NFL route family, with a
pre-registered, time-forward test against separation over expected (SOE).

Round 4, candidate 2. Run from the repo root:
    uv run python writeup/candidates/round_04/cand_2/analysis.py

Reads data/processed/*.parquet only (about one minute). Writes numbers.json, tables.json and figures/*.png
next to this file. The pre-registration (prereg.json) was committed before 2025-season outcomes were opened.

Definitions (identical code on drills and NFL snaps, 10 Hz x/y):
- velocity / acceleration from Savitzky-Golay derivatives of x/y (window 7, order 2);
- the break is the first >= 30 deg heading change (+-0.6 s) after 1.5 yd of travel, entered above 2.5 yd/s
  (round-3 detector, unchanged); the apex is the slowest frame within +-0.5 s of it;
- entry speed = peak speed in the 1.0 s before the apex;
- Speed Kept (SK, %) = apex speed / entry speed. z-scored within drill x Combine year (drills) and
  route family x position group (NFL);
- graded NFL snap = break apex at least 0.2 s before pass_forward.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DT = 0.1
RNG = np.random.default_rng(2027)
NBOOT = 2000
LEAD_MIN = 0.2  # pre-registered: apex >= 0.2 s before the throw

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10})
FIGW = 1600 / 150

NUM: dict[str, str] = {}
TABLES: dict = {}


def put(key, val, nd=2):
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key] = s
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_figheight())
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- kinematics
def kin(x, y):
    w = 7 if len(x) >= 7 else max(3, (len(x) // 2) * 2 - 1)
    vx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    vy = savgol_filter(y, w, 2, deriv=1, delta=DT)
    ax = savgol_filter(vx, w, 2, deriv=1, delta=DT)
    ay = savgol_filter(vy, w, 2, deriv=1, delta=DT)
    return vx, vy, ax, ay


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5):
    n = len(x)
    if n < 2 * h + 3:
        return None
    vx, vy, ax, ay = kin(x, y)
    s = np.hypot(vx, vy)
    head = np.degrees(np.arctan2(vx, vy)) % 360
    amag = np.hypot(ax, ay)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    cum = cum - cum[i0]
    turn = np.zeros(n)
    turn[h:n - h] = angdiff(head[:n - 2 * h], head[2 * h:])
    ok = np.zeros(n, bool)
    ok[max(i0 + h, h):min(i1, n - h)] = True
    ok &= cum >= min_path
    sin = np.array([s[max(0, i - h - 3):i - h + 1].max() if i - h >= 0 else 0.0 for i in range(n)])
    ok &= sin > entry_speed
    cand = np.where(ok & (turn >= min_turn))[0]
    if len(cand) == 0:
        return None
    j = cand[0]
    seg = np.arange(j, min(j + 8, n - h))
    seg = seg[ok[seg]] if ok[seg].any() else np.array([j])
    i = int(seg[np.argmax(turn[seg])])
    lo, hi = max(0, i - 5), min(n, i + 6)
    apex = lo + int(np.argmin(s[lo:hi]))
    a0 = max(0, apex - 10)
    iin = a0 + int(np.argmax(s[a0:apex + 1]))
    b1 = min(n, apex + 11)
    iout = apex + int(np.argmax(s[apex:b1]))
    pre, post = slice(max(0, i - 8), max(1, i - 3)), slice(min(n - 1, i + 3), min(n, i + 8))
    v1 = np.array([vx[pre].mean(), vy[pre].mean()])
    v2 = np.array([vx[post].mean(), vy[post].mean()])
    cosang = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return dict(i_break=i, i_apex=apex, i_in=iin, depth=float(cum[i]),
                angle=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                sk=float(100 * s[apex] / max(s[iin], 1e-6)), brr=float(amag[lo:hi].mean()))


def speed_trace(x, y):
    vx, vy, _, _ = kin(x, y)
    return np.hypot(vx, vy)


def clusters(cl):
    cl = np.asarray(cl)
    return [np.where(cl == c)[0] for c in pd.unique(cl)]


def boot_rho(a, b, cluster=None, n=NBOOT, q=(2.5, 97.5)):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    rho = spearmanr(a, b)[0]
    groups = clusters(np.asarray(cluster)[m]) if cluster is not None else None
    out = []
    for _ in range(n):
        if groups is None:
            ix = RNG.integers(0, len(a), len(a))
        else:
            ix = np.concatenate([groups[k] for k in RNG.integers(0, len(groups), len(groups))])
        out.append(spearmanr(a[ix], b[ix])[0])
    lo, hi = np.nanpercentile(out, q)
    return float(rho), float(lo), float(hi), int(len(a))


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


COMP = ["sk", "brr", "v_in", "v_apex", "v_out", "decel", "reaccel", "angle"]
COMP_LABEL = {"sk": "Speed Kept (SK)", "brr": "Break force (R3 BRR)", "v_in": "Entry speed", "v_apex": "Apex speed",
              "v_out": "Exit speed", "decel": "Decel into break", "reaccel": "Re-accel out", "angle": "Break angle"}

# ----------------------------------------------------------------------------- Combine drills
players = pd.read_parquet(P / "players.parquet")
cres = pd.read_parquet(P / "combine_results.parquet")
events = pd.read_parquet(P / "combine_events.parquet")
DRILL_FAMILY = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
                "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
                "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER",
                "CORNER_ROUTE": "CORNER"}
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
rows, dtraces = [], {}
for eid, d in frames.groupby("event_id", sort=False):
    pl_ = d[d.entity_type == "PLAYER"].sort_values("frame")
    bl = d[d.entity_type == "BALL"]
    x, y = pl_.x.to_numpy(), pl_.y.to_numpy()
    i1 = len(x)
    if len(bl):
        m = pl_[["frame", "x", "y"]].merge(bl[["frame", "x", "y"]], on="frame", suffixes=("", "_b"))
        c = m.frame[np.hypot(m.x - m.x_b, m.y - m.y_b) < 1.0]
        if len(c):
            i1 = int(np.searchsorted(pl_.frame.to_numpy(), c.iloc[0]))
    r = break_mechanics(x, y, 0, i1)
    if r:
        rows.append(dict(event_id=eid, **r))
        dtraces[eid] = (x, y)
cb = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group"]], on="event_id")
cb["family"] = cb.drill_name.map(DRILL_FAMILY)
put("n_drill_reps", len(ev)); put("n_drill_breaks", len(cb)); put("pct_drill_detected", 100 * len(cb) / len(ev), 0)
for m in COMP:
    cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
cpd = cb.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m + "_z" for m in COMP]].mean()
cplayer = cpd.groupby("nfl_id")[[m + "_z" for m in COMP]].mean()
cplayer["n_drills"] = cpd.groupby("nfl_id").size()
cplayer["n_reps"] = cb.groupby("nfl_id").size()
cplayer["sk_raw"] = cb.groupby("nfl_id").sk.median()
put("n_combine_players", len(cplayer))
put("median_drills_per_player", float(cplayer.n_drills.median()), 0)
HALF_A = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}

# ----------------------------------------------------------------------------- NFL snaps
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18)]
gp = gp.merge(players[["nfl_id", "position_group", "draft_year", "display_name", "draft_overall_pick"]], on="nfl_id")
gp = gp[gp.position_group.isin(["WR", "TE"])].copy()
gf = pd.read_parquet(P / "game_frames_sample.parquet")
gf = gf.merge(gp[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
grows, gtraces = [], {}
for k, d in gf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    d = d.sort_values("frame")
    e_ = d.event.fillna("").to_numpy()
    if not ((e_ == "ball_snap").any() and (e_ == "pass_forward").any()):
        continue
    i0 = int(np.argmax(e_ == "ball_snap"))
    ipf = int(np.argmax(e_ == "pass_forward"))
    base = dict(game_id=k[0], play_id=k[1], nfl_id=k[2], t_throw=(ipf - i0) * DT, ipf=ipf)
    r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
    if r:
        base.update(r)
        gtraces[k] = (d.x.to_numpy(), d.y.to_numpy(), i0, ipf)
    grows.append(base)
gb = pd.DataFrame(grows).merge(gp, on=["game_id", "play_id", "nfl_id"])
put("n_game_pass_snaps", len(gb)); put("n_game_players_all", gb.nfl_id.nunique())

# ---- SOE: context-only model fitted on 2023-24 NFL seasons, applied to 2025 (pre-registered)
gb["sep"] = gb.separation_at_pass_forward
gb["man"] = gb.team_coverage_type.fillna("").str.contains("MAN")
gb["slot"] = (gb.lined_up_position == "SLOT_WR").astype(float)
gb["inline"] = gb.lined_up_position.isin(["TE", "FB", "RB"]).astype(float)
gb["cush"] = gb.cushion.fillna(gb.cushion.median())
gb["cush_na"] = gb.cushion.isna().astype(float)
gb["pl"] = gb.pass_length.fillna(gb.pass_length.median())
gb["route"] = gb.route_ran.fillna("OTHER")
ROUTES = sorted(gb.route.unique())


def design(df):
    cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man.astype(float), df.slot, df.inline,
            df.t_throw, df.target.astype(float), df.down, df.yards_to_go]
    cols += [(df.route == r).astype(float) for r in ROUTES[1:]]
    return np.column_stack(cols)


dev = gb.season < 2025
beta_ctx = np.linalg.lstsq(design(gb[dev]), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
gb["xsep"] = design(gb) @ beta_ctx
gb["soe"] = gb.sep - gb.xsep
hold = ~dev
put("xsep_r2_2025", 1 - ((gb.sep[hold] - gb.xsep[hold]) ** 2).sum() / ((gb.sep[hold] - gb.sep[hold].mean()) ** 2).sum())
# variant used during development: context model cross-fitted by Combine class inside 2023-24
gb["soe_devcf"] = np.nan
for cls in (2023, 2024, 2025):
    tr, te = dev & (gb.draft_year != cls), dev & (gb.draft_year == cls)
    b = np.linalg.lstsq(design(gb[tr]), gb.loc[tr, "sep"].to_numpy(), rcond=None)[0]
    gb.loc[te, "soe_devcf"] = gb.loc[te, "sep"] - design(gb[te]) @ b

g = gb[gb.route_ran.isin(FAMILIES)].copy()
put("n_break_route_snaps", len(g))
put("pct_game_detected", 100 * g.brr.notna().mean(), 0)
g = g[g.brr.notna()].copy()
for m in COMP:
    g[m + "_z"] = g.groupby(["route_ran", "position_group"])[m].transform(zs)
g["lead"] = (g.ipf - g.i_apex) * DT
g["graded"] = g.lead >= LEAD_MIN
put("n_break_snaps", len(g)); put("n_break_players", g.nfl_id.nunique())
put("pct_after_throw", 100 * (g.lead < 0).mean(), 0)
put("pct_not_graded", 100 * (~g.graded).mean(), 0)
for f in FAMILIES:
    put(f"pct_not_graded_{f}", 100 * (~g.graded[g.route_ran == f]).mean(), 0)
q = g[g.graded].copy()
put("n_graded", len(q)); put("n_graded_players", q.nfl_id.nunique())
qd, qh = q[q.season < 2025], q[q.season == 2025]
put("n_dev", len(qd)); put("n_dev_players", qd.nfl_id.nunique())
put("n_hold", len(qh)); put("n_hold_players", qh.nfl_id.nunique())

# ----------------------------------------------------------------------------- pre-registered test
r_dev = boot_rho(qd.sk_z, qd.soe, cluster=qd.nfl_id)
put("dev_rho", r_dev[0]); put("dev_lo", r_dev[1]); put("dev_hi", r_dev[2])
put("dev_rho_cf", spearmanr(qd.sk_z, qd.soe_devcf)[0])
r_hold = boot_rho(qh.sk_z, qh.soe, cluster=qh.nfl_id)
put("hold_rho", r_hold[0]); put("hold_lo", r_hold[1]); put("hold_hi", r_hold[2])
r_hold90 = boot_rho(qh.sk_z, qh.soe, cluster=qh.nfl_id, q=(5, 95))
put("hold_lo90", r_hold90[1]); put("hold_hi90", r_hold90[2])
# within-player permutation (SK shuffled within player), one-sided
sk_h, soe_h, pid_h = qh.sk_z.to_numpy(), qh.soe.to_numpy(), qh.nfl_id.to_numpy()
grp = clusters(pid_h)
obs = spearmanr(sk_h, soe_h)[0]
null = []
for _ in range(NBOOT):
    sh = sk_h.copy()
    for ix in grp:
        sh[ix] = sh[RNG.permutation(ix)]
    null.append(spearmanr(sh, soe_h)[0])
put("hold_p", (1 + np.sum(np.array(null) >= obs)) / (1 + len(null)), 2)
put("primary_pass", int(obs > 0 and np.mean(np.array(null) >= obs) < 0.05 and r_hold90[1] > 0))
r_all = boot_rho(q.sk_z, q.soe, cluster=q.nfl_id)
put("all_rho", r_all[0]); put("all_lo", r_all[1]); put("all_hi", r_all[2])

fam_rows = []
for f in FAMILIES:
    for lab, d in (("2023-24 (development)", qd), ("2025 (held out)", qh)):
        dd = d[d.route_ran == f]
        r = boot_rho(dd.sk_z, dd.soe, cluster=dd.nfl_id)
        fam_rows.append(dict(family=f, period=lab, rho=r[0], lo=r[1], hi=r[2], n=r[3]))
        key = "dev" if lab.startswith("2023") else "hold"
        put(f"fam_{key}_{f}", r[0]); put(f"fam_{key}_{f}_n", r[3])
fam = pd.DataFrame(fam_rows)
put("n_fam_dev_positive", int((fam[fam.period.str.startswith("2023")].rho > 0).sum()))
put("n_fam_hold_positive", int((fam[fam.period.str.startswith("2025")].rho > 0).sum()))
# out + corner combined (post hoc; basis for the 2026 hypothesis only)
oc_d = qd[qd.route_ran.isin(["OUT", "CORNER"])]
oc_h = qh[qh.route_ran.isin(["OUT", "CORNER"])]
r_ocd = boot_rho(oc_d.sk_z, oc_d.soe, cluster=oc_d.nfl_id)
r_och = boot_rho(oc_h.sk_z, oc_h.soe, cluster=oc_h.nfl_id)
put("oc_dev", r_ocd[0]); put("oc_dev_n", r_ocd[3]); put("oc_hold", r_och[0]); put("oc_hold_lo", r_och[1])
put("oc_hold_hi", r_och[2]); put("oc_hold_n", r_och[3])
# effect size in yards for out+corner, all seasons, top vs bottom SK quintile
oc = q[q.route_ran.isin(["OUT", "CORNER"])].copy()
full, low = oc[oc.sk >= 99.999], oc[oc.sk < 80]
put("oc_full_n", len(full)); put("oc_low_n", len(low))
put("oc_full_soe", full.soe.mean()); put("oc_low_soe", low.soe.mean())
put("oc_gap_in", 36 * (full.soe.mean() - low.soe.mean()), 0)
gi_oc = clusters(oc.nfl_id)
sv, skv = oc.soe.to_numpy(), oc.sk.to_numpy()
bs_oc = []
for _ in range(NBOOT):
    ix = np.concatenate([gi_oc[k] for k in RNG.integers(0, len(gi_oc), len(gi_oc))])
    a_, b_ = sv[ix][skv[ix] >= 99.999], sv[ix][skv[ix] < 80]
    bs_oc.append(36 * (a_.mean() - b_.mean()))
put("oc_gap_lo", np.percentile(bs_oc, 2.5), 0); put("oc_gap_hi", np.percentile(bs_oc, 97.5), 0)
bs_sl = []
for _ in range(NBOOT):
    ix = np.concatenate([gi_oc[k] for k in RNG.integers(0, len(gi_oc), len(gi_oc))])
    bs_sl.append(36 * 10 * np.polyfit(skv[ix], sv[ix], 1)[0])
put("oc_slope_in", 36 * 10 * np.polyfit(skv, sv, 1)[0], 1)
put("oc_slope_lo", np.percentile(bs_sl, 2.5), 1); put("oc_slope_hi", np.percentile(bs_sl, 97.5), 1)
put("oc_n_all", len(oc))
# within-player version (all graded snaps)
q["sk_w"] = q.sk_z - q.groupby("nfl_id").sk_z.transform("mean")
q["soe_w"] = q.soe - q.groupby("nfl_id").soe.transform("mean")
r_w = boot_rho(q.sk_w, q.soe_w, cluster=q.nfl_id)
put("within_rho", r_w[0]); put("within_lo", r_w[1]); put("within_hi", r_w[2])
put("pct_cap_all", 100 * (q.sk >= 99.999).mean(), 0)
for f in FAMILIES:
    put(f"pct_cap_{f}", 100 * (q[q.route_ran == f].sk >= 99.999).mean(), 0)
oc_w = q[q.route_ran.isin(["OUT", "CORNER"])]
r_ocw = boot_rho(oc_w.sk_w, oc_w.soe_w, cluster=oc_w.nfl_id)
put("oc_within", r_ocw[0]); put("oc_within_lo", r_ocw[1]); put("oc_within_hi", r_ocw[2])
oc_m = oc_w.groupby("nfl_id").agg(sk=("sk_z", "mean"), soe=("soe", "mean"), n=("sk", "size"))
oc_m = oc_m[oc_m.n >= 3]
r_ocb = boot_rho(oc_m.sk, oc_m.soe)
put("oc_between", r_ocb[0]); put("oc_between_n", r_ocb[3])
# typical raw SK by family (graded snaps)
for f in FAMILIES:
    put(f"sk_med_game_{f}", g[g.route_ran == f].sk.median(), 0)
    put(f"sk_med_drill_{f}", cb[cb.family == f].sk.median(), 0)

# ---- timing diagnostic: round-3 BRR man-minus-zone, graded vs after-throw snaps (all seasons)
tim_rows = []
for lab, d in (("Break before the throw", g[g.graded]), ("Break at or after the throw", g[~g.graded])):
    for cov, mk in (("man", d.man), ("zone", ~d.man)):
        dd = d[mk]
        r = boot_rho(dd.brr_z, dd.soe, cluster=dd.nfl_id)
        tim_rows.append(dict(subset=lab, coverage=cov, rho=r[0], lo=r[1], hi=r[2], n=r[3]))
tim = pd.DataFrame(tim_rows)
put("brr_pre_man", tim.rho[0]); put("brr_pre_zone", tim.rho[1]); put("brr_post_man", tim.rho[2])
put("brr_post_zone", tim.rho[3]); put("brr_post_zone_lo", tim.lo[3]); put("brr_post_zone_hi", tim.hi[3])
put("brr_pre_zone_lo", tim.lo[1]); put("brr_pre_zone_hi", tim.hi[1])
hz = g[(~g.graded) & (g.route_ran == "HITCH")]
put("pct_hitch_post", 100 * len(hz) / (g.route_ran == "HITCH").sum(), 0)
lead_bins = pd.cut(g.lead, [-5, 0, LEAD_MIN - 1e-9, 1, 2, 20], labels=["after throw", "0-0.2 s", "0.2-1 s", "1-2 s", ">2 s"])
lb = g.groupby(lead_bins, observed=False).agg(n=("soe", "size"), soe=("soe", "mean"))

# ----------------------------------------------------------------------------- reliability (all break snaps)
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
rel_rows = []
for m in COMP:
    ha = cpd[cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hb = cpd[~cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hh = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
    rc = boot_rho(hh.a, hh.b)
    gh = g.groupby(["nfl_id", "half"])[m + "_z"].mean().unstack().dropna()
    rg = boot_rho(gh[0], gh[1])
    rel_rows.append(dict(comp=m, c_rho=rc[0], c_lo=rc[1], c_hi=rc[2], c_n=rc[3],
                         g_rho=rg[0], g_lo=rg[1], g_hi=rg[2], g_n=rg[3]))
    put(f"rel_c_{m}", rc[0]); put(f"rel_g_{m}", rg[0])
    if m == "sk":
        put("rel_c_sk_lo", rc[1]); put("rel_c_sk_hi", rc[2]); put("rel_c_n", rc[3])
        put("rel_g_sk_lo", rg[1]); put("rel_g_sk_hi", rg[2]); put("rel_g_n", rg[3])
rel = pd.DataFrame(rel_rows)
put("max_rel_g", rel.g_rho.max()); put("max_rel_c", rel.c_rho.max())
put("snaps_per_player", g.groupby("nfl_id").size().mean(), 1)
# ICC(1) of SK in NFL snaps, one-way ANOVA
def icc1(df, col):
    k = df.groupby("nfl_id").size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby("nfl_id")[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby("nfl_id")[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


icc = icc1(g, "sk_z")
pids = g.nfl_id.unique()
gi = {p: d for p, d in g.groupby("nfl_id")}
icc_b = [icc1(pd.concat([gi[p].assign(nfl_id=i) for i, p in enumerate(RNG.choice(pids, len(pids)))]), "sk_z")
         for _ in range(500)]
put("icc_sk", icc, 3); put("icc_sk_hi", np.percentile(icc_b, 97.5), 3)
put("snaps_for_07_icc", int(np.ceil(0.7 * (1 - max(np.percentile(icc_b, 97.5), 1e-3)) /
                                    (max(np.percentile(icc_b, 97.5), 1e-3) * 0.3))))

# ----------------------------------------------------------------------------- Combine -> NFL translation
gplayer = g.groupby("nfl_id").agg(sk_game=("sk_z", "mean"), n_snaps=("sk", "size"), sk_game_raw=("sk", "median"),
                                  sk_game_sd=("sk_z", "std"))
X = cplayer.join(gplayer, how="inner").join(players.set_index("nfl_id")[["display_name", "draft_year",
                                                                          "draft_overall_pick", "position_group"]])
tr = boot_rho(X.sk_z, X.sk_game)
put("tr_rho", tr[0]); put("tr_lo", tr[1]); put("tr_hi", tr[2]); put("tr_n", tr[3])
cfam = cpd.groupby(["nfl_id", "family"]).sk_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).sk_z.mean()
gfam.index.names = ["nfl_id", "family"]
jf = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
fr = boot_rho(jf.d, jf.g, cluster=jf.nfl_id)
put("fam_tr_rho", fr[0]); put("fam_tr_lo", fr[1]); put("fam_tr_hi", fr[2]); put("fam_tr_n", fr[3])
# Combine SK vs NFL SOE on graded snaps (player level)
ps = q.groupby("nfl_id").soe.mean().rename("soe_p")
Y = cplayer.join(ps, how="inner")
r_cs = boot_rho(Y.sk_z, Y.soe_p)
put("comb_soe_rho", r_cs[0]); put("comb_soe_lo", r_cs[1]); put("comb_soe_hi", r_cs[2]); put("comb_soe_n", r_cs[3])

# ---- movement mismatch, drill vs game (all break snaps)
prof = []
for f in FAMILIES:
    dg, dc = g[g.route_ran == f], cb[cb.family == f]
    for var in ("v_in", "angle", "sk"):
        for lab, d in (("Combine drill", dc), ("NFL snap", dg)):
            grp_ = clusters(d.nfl_id)
            vals = d[var].to_numpy()
            bs = [np.median(vals[np.concatenate([grp_[k] for k in RNG.integers(0, len(grp_), len(grp_))])])
                  for _ in range(400)]
            prof.append(dict(family=f, var=var, src=lab, med=float(np.median(vals)),
                             lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=len(d)))
prof = pd.DataFrame(prof)
for f in FAMILIES:
    for var, nd in (("v_in", 1), ("angle", 0)):
        put(f"{var}_drill_{f}", prof[(prof.family == f) & (prof["var"] == var) & (prof.src == "Combine drill")].med.iloc[0], nd)
        put(f"{var}_game_{f}", prof[(prof.family == f) & (prof["var"] == var) & (prof.src == "NFL snap")].med.iloc[0], nd)
put("vin_drill_all", cb.v_in.median(), 1); put("vin_game_all", g.v_in.median(), 1)
put("sk_drill_all", cb.sk.median(), 0); put("sk_game_all", g.sk.median(), 0)

# ----------------------------------------------------------------------------- baselines without tracking
cr = cres.set_index("nfl_id")
pb = players.set_index("nfl_id")[["draft_overall_pick", "position_group", "draft_year"]].join(
    cr[["forty", "combine_weight", "ngs_athleticism_score"]])
pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
pb["te"] = (pb.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score"):
    pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs)
    pb[c + "_s"] = pb[c + "_s"].fillna(0.0)
BASE = ["logpick", "forty_s", "combine_weight_s", "ngs_athleticism_score_s", "te"]
put("n_missing_forty", int(pb.loc[pb.position_group.isin(["WR", "TE"]) & pb.forty.isna()].shape[0]))

# (a) snap level: raw separation on graded 2025 snaps, models fit on graded 2023-24 snaps
qq = q.join(pb[BASE], on="nfl_id")


def fit_pred(cols_fn, tr_, te_):
    A, B = cols_fn(tr_), cols_fn(te_)
    b = np.linalg.lstsq(A, tr_.sep.to_numpy(), rcond=None)[0]
    return B @ b


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


trq, teq = qq[qq.season < 2025], qq[qq.season == 2025]
f_ctx = design
f_base = lambda d: np.column_stack([design(d), d[BASE].to_numpy()])  # noqa: E731
f_sk = lambda d: np.column_stack([design(d), d[BASE].to_numpy(), d.sk_z.to_numpy()])  # noqa: E731
y = teq.sep.to_numpy()
snap_r2 = {k: r2(y, fit_pred(fn, trq, teq)) for k, fn in (("ctx", f_ctx), ("base", f_base), ("sk", f_sk))}
gi_ = clusters(teq.nfl_id)
pc, pbb, ps_ = fit_pred(f_ctx, trq, teq), fit_pred(f_base, trq, teq), fit_pred(f_sk, trq, teq)
dlt = []
for _ in range(NBOOT):
    ix = np.concatenate([gi_[k] for k in RNG.integers(0, len(gi_), len(gi_))])
    dlt.append(r2(y[ix], ps_[ix]) - r2(y[ix], pbb[ix]))
put("snap_r2_ctx", snap_r2["ctx"], 3); put("snap_r2_base", snap_r2["base"], 3); put("snap_r2_sk", snap_r2["sk"], 3)
put("snap_dr2", snap_r2["sk"] - snap_r2["base"], 3)
put("snap_dr2_lo", np.percentile(dlt, 2.5), 3); put("snap_dr2_hi", np.percentile(dlt, 97.5), 3)
put("snap_dr2_base", snap_r2["base"] - snap_r2["ctx"], 3)

# (b) player-season level, out of class: routes-weighted season separation and YPRR
ss = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss[ss.position_group.isin(["WR", "TE"]) & (ss.routes >= 150)].copy()
ss["w"] = ss.routes
agg = ss.groupby("nfl_id").apply(lambda d: pd.Series({
    "sep": np.average(d.avg_separation, weights=d.w), "yprr": np.average(d.yards_per_route, weights=d.w),
    "routes": d.routes.sum()}), include_groups=False)
S = agg.join(pb, how="inner").join(cplayer[["sk_z"]], how="inner")
put("n_season_players", len(S))


def ridge_loco(df, cols, ycol, lam=1.0):
    pred = pd.Series(np.nan, index=df.index)
    for cls in (2023, 2024, 2025):
        trn, tst = df[df.draft_year != cls], df[df.draft_year == cls]
        if len(tst) == 0:
            continue
        mu, sd = trn[cols].mean(), trn[cols].std().replace(0, 1)
        A = np.column_stack([np.ones(len(trn)), ((trn[cols] - mu) / sd).to_numpy()])
        Bm = np.column_stack([np.ones(len(tst)), ((tst[cols] - mu) / sd).to_numpy()])
        pen = lam * np.eye(A.shape[1]); pen[0, 0] = 0
        b = np.linalg.solve(A.T @ A + pen, A.T @ trn[ycol].to_numpy())
        pred[tst.index] = Bm @ b
    return pred


season_rows = []
for yc, lab in (("sep", "Season separation (yd)"), ("yprr", "Yards per route run")):
    p0 = ridge_loco(S, BASE, yc)
    p1 = ridge_loco(S, BASE + ["sk_z"], yc)
    yy = S[yc]
    d0, d1 = r2(yy, p0), r2(yy, p1)
    bs = []
    idx = np.arange(len(S))
    for _ in range(NBOOT):
        ix = RNG.integers(0, len(S), len(S))
        bs.append(r2(yy.iloc[ix], p1.iloc[ix]) - r2(yy.iloc[ix], p0.iloc[ix]))
    season_rows.append(dict(outcome=lab, base=d0, plus=d1, d=d1 - d0, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    put(f"season_{yc}_base", d0); put(f"season_{yc}_plus", d1); put(f"season_{yc}_d", d1 - d0)
    put(f"season_{yc}_lo", np.percentile(bs, 2.5)); put(f"season_{yc}_hi", np.percentile(bs, 97.5))

# ----------------------------------------------------------------------------- named players
named = X[(X.n_reps >= 4) & (X.n_snaps >= 10)].copy()
named["g_lo"] = named.sk_game - 1.645 * named.sk_game_sd / np.sqrt(named.n_snaps)
named["g_hi"] = named.sk_game + 1.645 * named.sk_game_sd / np.sqrt(named.n_snaps)
named = named.sort_values("sk_z", ascending=False)
put("n_named_pool", len(named))
top, bot = named.head(5), named.tail(5)
named_rows = []
for tag, d in (("Top Combine SK", top), ("Bottom Combine SK", bot)):
    for pid, r_ in d.iterrows():
        named_rows.append(dict(tag=tag, name=r_.display_name, pos=r_.position_group, cls=int(r_.draft_year),
                               pick=None if pd.isna(r_.draft_overall_pick) else int(r_.draft_overall_pick),
                               c_sk=round(float(r_.sk_z), 2), c_raw=round(float(r_.sk_raw), 0), reps=int(r_.n_reps),
                               g_sk=round(float(r_.sk_game), 2), g_lo=round(float(r_.g_lo), 2),
                               g_hi=round(float(r_.g_hi), 2), n=int(r_.n_snaps)))
for i, r_ in enumerate(named_rows):
    for k in ("c_sk", "g_sk", "g_lo", "g_hi"):
        put(f"named_{i}_{k}", r_[k])
    put(f"named_{i}_n", r_["n"]); put(f"named_{i}_reps", r_["reps"])
put("named_top_mean_game", top.sk_game.mean()); put("named_bot_mean_game", bot.sk_game.mean())
put("named_top_mean_comb", top.sk_z.mean()); put("named_bot_mean_comb", bot.sk_z.mean())
put("named_pool_halfwidth", float((named.g_hi - named.g_lo).median() / 2))
TABLES["named"] = named_rows
TABLES["season"] = season_rows
TABLES["fam"] = fam.round(3).to_dict("records")
TABLES["rel"] = rel.round(3).to_dict("records")
TABLES["timing"] = tim.round(3).to_dict("records")
TABLES["lead_bins"] = lb.reset_index().astype({"lead": str}).round(3).to_dict("records")

# ============================================================================= FIGURES
C_D, C_G = OI["blue"], OI["orange"]

# ---- Fig 1: the metric on one rule-chosen player (most graded NFL OUT snaps with a Combine speed-out rep)
so = cb[cb.drill_name == "SPEED_OUT_ROUTE_LEFT"]
outs = q[(q.route_ran == "OUT") & q.nfl_id.isin(so.nfl_id)]
pid1 = outs.groupby("nfl_id").size().sort_values(ascending=False).index[0]
nm1 = players.set_index("nfl_id").display_name[pid1]
put("fig1_n_out", int((outs.nfl_id == pid1).sum()))
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})


def rot_path(x, y_, apex):
    lo_, hi_ = max(0, apex - 12), min(len(x), apex + 11)
    px, py = x[lo_:hi_] - x[apex], y_[lo_:hi_] - y_[apex]
    hx, hy = x[apex - 3] - x[lo_], y_[apex - 3] - y_[lo_]
    th = np.arctan2(hx, hy)
    rx = px * np.cos(th) - py * np.sin(th)
    ry = px * np.sin(th) + py * np.cos(th)
    if rx[-1] < 0:
        rx = -rx
    return rx, ry


ax, axp = axs
for _, r_ in so[so.nfl_id == pid1].iterrows():
    x, y_ = dtraces[r_.event_id]
    s = speed_trace(x, y_)
    t = np.arange(len(s)) - r_.i_apex
    ax.plot(t * DT, s, color=C_D, lw=2.6, label=f"Combine speed-out rep: SK {r_.sk:.0f}%", zorder=3)
    ax.plot((r_.i_in - r_.i_apex) * DT, r_.v_in, "o", color=C_D, ms=8, zorder=4)
    ax.plot(0, r_.v_apex, "o", mfc="white", mec=C_D, mew=2, ms=8, zorder=4)
    ax.annotate(f"entry {r_.v_in:.1f} yd/s", ((r_.i_in - r_.i_apex) * DT, r_.v_in), xytext=(-1.15, 7.7), fontsize=9,
                color=C_D, arrowprops=dict(arrowstyle="-", color=C_D, lw=0.8))
    ax.annotate(f"apex {r_.v_apex:.1f} yd/s", (0, r_.v_apex), xytext=(0.15, 7.7), fontsize=9, color=C_D,
                arrowprops=dict(arrowstyle="-", color=C_D, lw=0.8))
    put("fig1_drill_sk", r_.sk, 0)
    rx, ry = rot_path(x, y_, int(r_.i_apex))
    axp.plot(rx, ry, color=C_D, lw=2.6, zorder=3)
    axp.plot(0, 0, "o", mfc="white", mec=C_D, mew=2, ms=7, zorder=4)
sks = []
for _, r_ in outs[outs.nfl_id == pid1].iterrows():
    x, y_, i0, ipf = gtraces[(r_.game_id, r_.play_id, r_.nfl_id)]
    s = speed_trace(x, y_)
    t = np.arange(len(s)) - r_.i_apex
    m = (t >= -12) & (t <= 10)
    ax.plot(t[m] * DT, s[m], color=C_G, lw=1.1, alpha=0.75)
    if ipf < len(s) and -12 <= ipf - r_.i_apex <= 10:
        ax.plot((ipf - r_.i_apex) * DT, s[ipf], "v", color=OI["verm"], ms=6)
    rx, ry = rot_path(x, y_, int(r_.i_apex))
    axp.plot(rx, ry, color=C_G, lw=1.1, alpha=0.75)
    sks.append(r_.sk)
ax.plot([], [], color=C_G, lw=1.1, label=f"his graded NFL out routes: median SK {np.median(sks):.0f}% (n={len(sks)})")
ax.plot([], [], "v", color=OI["verm"], label="ball thrown")
put("fig1_game_sk", np.median(sks), 0)
ax.axvline(0, color=OI["grey"], ls=":", lw=1)
ax.set_xlabel("seconds from break apex (slowest point of the break)")
ax.set_ylabel("speed (yd/s)")
ax.set_ylim(0, 8.4)
ax.set_title(f"Speed Kept = apex speed / entry speed ({nm1}, out route)", loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower left")
ax.set_xlim(-1.2, 1.0)
axp.set_aspect("equal")
axp.set_xlabel("yards across (break to the right)")
axp.set_ylabel("yards upfield (0 = apex)")
axp.set_title("Same reps, path through the break", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "fig1_metric.png")

# ---- Fig 2: timing
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]
bins = np.arange(-1.0, 3.01, 0.1)
for f, c in zip(FAMILIES, [OI["sky"], OI["orange"], OI["verm"], OI["green"], OI["purple"]]):
    ax.hist(g.loc[g.route_ran == f, "lead"].clip(-1, 3), bins=bins, histtype="step", lw=1.6, color=c,
            label=f"{f.title()} ({NUM['pct_not_graded_' + f]}% ungraded)")
ax.axvspan(-1, LEAD_MIN, color=OI["grey"], alpha=0.15)
ax.set_xlabel("seconds from break apex to throw (below 0: ball already out)")
ax.set_ylabel("NFL snaps")
ax.set_title(f"{NUM['pct_not_graded']}% of breaks finish < 0.2 s before the throw (shaded)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax = axs[1]
ypos = {("Break before the throw", "man"): 3, ("Break before the throw", "zone"): 2,
        ("Break at or after the throw", "man"): 1, ("Break at or after the throw", "zone"): 0}
for _, r_ in tim.iterrows():
    yy_ = ypos[(r_.subset, r_.coverage)]
    c = OI["blue"] if r_.coverage == "man" else OI["verm"]
    ax.errorbar(r_.rho, yy_, xerr=[[r_.rho - r_.lo], [r_.hi - r_.rho]], fmt="o", color=c, capsize=3)
    ax.text(r_.hi + 0.02, yy_, f"n={r_.n}", va="center", fontsize=8)
ax.set_yticks([3, 2, 1, 0], ["before throw, man", "before throw, zone", "after throw, man", "after throw, zone"])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("rho, round-3 break force vs separation over expected")
ax.set_title("Round 3's zone penalty sat in post-throw breaks", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "fig2_timing.png")

# ---- Fig 3: pre-registered test
fig, ax = plt.subplots(figsize=(FIGW, 4.6))
labels = ["All five families"] + [f.title() for f in FAMILIES]
for k, (lab, c, off) in enumerate((("2023-24 (development)", C_D, 0.17), ("2025 (held out)", C_G, -0.17))):
    rows_ = [dict(rho=r_dev[0], lo=r_dev[1], hi=r_dev[2], n=r_dev[3]) if k == 0 else
             dict(rho=r_hold[0], lo=r_hold[1], hi=r_hold[2], n=r_hold[3])]
    rows_ += fam[fam.period == lab][["rho", "lo", "hi", "n"]].to_dict("records")
    for i, r_ in enumerate(rows_):
        yv = len(labels) - 1 - i + off
        ax.errorbar(r_["rho"], yv, xerr=[[r_["rho"] - r_["lo"]], [r_["hi"] - r_["rho"]]], fmt="o" if i else "D",
                    color=c, capsize=3, ms=7 if i == 0 else 5, label=lab if i == 0 else None)
        ax.text(0.62, yv, f"n={r_['n']}", va="center", fontsize=8, color=c)
ax.set_yticks(range(len(labels)), labels[::-1])
ax.axvline(0, color="black", lw=0.8)
ax.axhline(len(labels) - 1.5, color=OI["grey"], lw=0.6, ls=":")
ax.set_xlim(-0.65, 0.72)
ax.set_xlabel("rho, Speed Kept vs separation over expected (95% CI, players resampled)")
ax.set_title(f"Pre-registered test failed: 2025 rho = {NUM['hold_rho']} (one-sided p = {NUM['hold_p']})", loc="left",
             fontsize=11)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
save(fig, "fig3_prereg.png")

# ---- Fig 4: reliability
fig, ax = plt.subplots(figsize=(FIGW, 4.4))
order = rel.comp.tolist()
for i, r_ in rel.iterrows():
    yv = len(order) - 1 - i
    ax.errorbar(r_.c_rho, yv + 0.15, xerr=[[r_.c_rho - r_.c_lo], [r_.c_hi - r_.c_rho]], fmt="s", color=C_D, capsize=3,
                label="Combine: two disjoint drill sets" if i == 0 else None)
    ax.errorbar(r_.g_rho, yv - 0.15, xerr=[[r_.g_rho - r_.g_lo], [r_.g_hi - r_.g_rho]], fmt="o", color=C_G, capsize=3,
                label=f"NFL: alternate break snaps ({NUM['snaps_per_player']} per player)" if i == 0 else None)
ax.set_yticks(range(len(order)), [COMP_LABEL[c] for c in order][::-1])
ax.axvline(0, color="black", lw=0.8)
ax.axvline(0.5, color=OI["grey"], lw=0.8, ls="--")
ax.text(0.51, len(order) - 0.6, "0.5: usable for\nranking players", fontsize=8, color=OI["grey"], va="top")
ax.set_xlabel("split-half rho (does the player repeat himself?)")
ax.set_title("No break component is a stable player trait at either venue", loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
save(fig, "fig4_reliability.png")

# ---- Fig 5: movement mismatch
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 4.0))
for ax, (var, lab) in zip(axs, (("v_in", "entry speed (yd/s)"), ("angle", "break angle (deg)"), ("sk", "Speed Kept (%)"))):
    for k, (src, c, off) in enumerate((("Combine drill", C_D, -0.15), ("NFL snap", C_G, 0.15))):
        d = prof[(prof["var"] == var) & (prof.src == src)].set_index("family").loc[FAMILIES]
        xv = np.arange(len(FAMILIES)) + off
        ax.errorbar(xv, d.med, yerr=[d.med - d.lo, d.hi - d.med], fmt="o", color=c, capsize=3, label=src)
    ax.set_xticks(range(len(FAMILIES)), [f.title() for f in FAMILIES], rotation=30)
    ax.set_title(lab, loc="left", fontsize=11)
axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("Drill breaks are entered faster, and drill hitches turn far sharper, than NFL breaks (medians, 95% CI)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "fig5_mismatch.png")

NUM["abs_values"] = sorted({v.lstrip("-") for v in NUM.values() if isinstance(v, str)})
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "tables.json").write_text(json.dumps(TABLES, indent=1, default=str))
print(json.dumps(NUM, indent=1))
print(json.dumps(TABLES["named"], indent=1))
print(json.dumps(TABLES["season"], indent=1, default=str))
print(TABLES["lead_bins"])
