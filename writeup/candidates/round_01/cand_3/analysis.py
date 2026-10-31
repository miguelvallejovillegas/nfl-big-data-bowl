"""Burst Translation: does a player's Combine first-second speed profile show up on NFL snaps of the
same movement type, and does the gap (translation) explain production beyond draft slot?

Reads data/processed/*.parquet (and reuses analysis/insights.py:load for career outcomes), writes
numbers.json, tables.json and figures/*.png next to this file.

Usage: uv run python writeup/candidates/round_01/cand_3/analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROC = ROOT / "data" / "processed"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))
import insights  # noqa: E402

RNG = np.random.default_rng(2027)
DT = 0.1
T = 16                 # frames kept after onset (0.0-1.5 s)
BURST = 11             # first-second window: frames 0..10 (0.0-1.0 s)
ON, PRE = 1.0, 0.5     # onset threshold (yd/s) and back-off threshold
GROUPS = ["DL", "OL", "WR", "TE", "DB"]
FOCUS = "DL"
NBOOT = 2000
WIDE_ALIGN = {"EDGE", "OLB", "T", "WR", "CB"}   # outside alignments per group (share used as a control)
MATCHED = {
    "DL": ["PASS_RUSH_DRILL"],
    "OL": ["PASS_RUSH_DROP", "PASS_PRO_MIRROR_DRILL"],
    "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT",
           "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT"],
    "TE": ["CORNER_ROUTE", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "FLAT_ROUTE_LEFT", "WHEEL_ROUTE_LEFT"],
    "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "LINE_DRILL"],
}
GAME_TYPE = {"DL": "pass-rush snaps", "OL": "pass-block snaps", "WR": "routes", "TE": "routes", "DB": "coverage snaps"}
DRILL_LABEL = {"DL": "Pass-rush drill", "OL": "Pass-rush drop + mirror", "WR": "8 route drills", "TE": "5 route drills",
               "DB": "Backpedal drills (3)"}
OUTCOME_LABEL = {"DL": "Pressure rate", "OL": "Pressure rate allowed (sign flipped)", "WR": "Yards per route",
                 "TE": "Yards per route", "DB": "Solo tackles per 100 snaps"}
# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
GCOL = {"DL": OI["red"], "OL": OI["orange"], "WR": OI["blue"], "TE": OI["green"], "DB": OI["purple"]}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}
ABS: set = set()


def put(key: str, value, fmt: str = "{:.2f}") -> str:
    """Store a formatted number for the writeup and return it."""
    s = fmt.format(value) if not isinstance(value, str) else value
    NUM[key] = s
    ABS.add(s.lstrip("-"))
    return s


def log(m: str) -> None:
    print(m, flush=True)


# ----------------------------------------------------------------------------- curves
def onset_curve(s: np.ndarray, start: int = 0):
    """Speed for T frames from movement onset (first frame >= ON yd/s, backed up <=3 frames while >= PRE)."""
    s = np.asarray(s, float)[start:]
    if not len(s) or s[0] >= ON:      # already moving at clip start / snap: onset not observed
        return None
    hit = np.flatnonzero(s >= ON)
    if not len(hit):
        return None
    i = j = int(hit[0])
    while j > 0 and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    c = s[j:j + T]
    return c if len(c) == T else None


def zscore(v: pd.Series) -> pd.Series:
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else v * np.nan


def build_curves(p: pd.DataFrame):
    ev = pd.read_parquet(PROC / "combine_events.parquet")
    cf = pd.read_parquet(PROC / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
    cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    sp = dict(tuple(cf.groupby("event_id").s))
    rows = []
    for g, drills in MATCHED.items():
        e = ev[(ev.position_group == g) & ev.drill_name.isin(drills)]
        for r in e.itertuples():
            c = onset_curve(sp[r.event_id].values)
            if c is not None:
                rows.append((g, r.nfl_id, r.drill_name, r.draft_year, r.attempt_no, *c))
    D = pd.DataFrame(rows, columns=["grp", "nfl_id", "drill", "yr", "att"] + [f"c{i}" for i in range(T)])
    n_drill_events = int(ev[ev.apply(lambda r: r.drill_name in MATCHED.get(r.position_group, []), axis=1)].shape[0])
    # Combine top speed from the 40 (best attempt), year-standardized
    f40 = ev[ev.drill_type == "FORTY_YARD_DASH"]
    top40 = f40.groupby(["nfl_id", "draft_year"]).top_speed.max().reset_index()
    top40["top40_z"] = top40.groupby("draft_year").top_speed.transform(zscore)
    t40_pairs = f40.sort_values("attempt_no").groupby("nfl_id").top_speed.apply(lambda v: v.values[:2] if len(v) >= 2 else None).dropna()

    gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s"])
    gf = gf.sort_values(["game_id", "play_id", "nfl_id", "frame"])
    k = ["game_id", "play_id", "nfl_id"]
    snapf = gp.set_index(k).snap_frame
    grows, peaks = [], []
    for key, d in gf.groupby(k, sort=False):
        sf = snapf.get(key, np.nan)
        if np.isnan(sf):
            continue
        s = d.s.values
        st = int(np.searchsorted(d.frame.values, sf))
        peaks.append((*key, float(s[st:].max()) if len(s) > st else np.nan))
        c = onset_curve(s, st)
        if c is not None:
            grows.append((*key, *c))
    G = pd.DataFrame(grows, columns=k + [f"c{i}" for i in range(T)])
    PK = pd.DataFrame(peaks, columns=k + ["peak"])
    meta = gp[k + ["season", "lined_up_position", "expected_points_added", "player_get_off"]].copy()
    meta["ae"] = meta.expected_points_added.abs()
    meta["rk"] = meta.groupby("nfl_id").ae.rank(ascending=False, method="first")
    meta["nn"] = meta.groupby("nfl_id").ae.transform("size")
    # ETL drew 6 top-|EPA| plays + 6 random plays per player; the 6 lowest-|EPA| of 12 are the random half
    meta["random_half"] = (meta.nn < 12) | (meta.rk > 6)
    meta = meta.merge(p[["nfl_id", "position_group"]], on="nfl_id")
    G = G.merge(meta, on=k)
    PK = PK.merge(meta, on=k)
    G["grp"] = G.position_group
    PK["grp"] = PK.position_group
    return D, G, PK, top40, t40_pairs, gp, n_drill_events


# ----------------------------------------------------------------------------- stats helpers
def partial_spearman(x, y, Z: pd.DataFrame):
    df = pd.concat([pd.Series(x, name="x"), pd.Series(y, name="y"), Z], axis=1).dropna()
    R = df.rank()
    zc = [c for c in R.columns if c not in ("x", "y")]
    A = np.c_[np.ones(len(R)), R[zc].values]
    rx = R.x - A @ np.linalg.lstsq(A, R.x, rcond=None)[0]
    ry = R.y - A @ np.linalg.lstsq(A, R.y, rcond=None)[0]
    r, pv = stats.pearsonr(rx, ry)
    boot = []
    n = len(R)
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        a, b = rx.values[i], ry.values[i]
        if a.std() > 0 and b.std() > 0:
            boot.append(np.corrcoef(a, b)[0, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(r), float(pv), int(n), float(lo), float(hi)


def spearman_ci(x, y):
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    r = stats.spearmanr(x, y)[0]
    n = len(x)
    b = [stats.spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, n, n) for _ in range(NBOOT))]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(r), float(lo), float(hi), int(n)


def icc1(df: pd.DataFrame, id_col: str, val: str) -> float:
    """One-way random-effects ICC(1): reliability of a single rep."""
    d = df[[id_col, val]].dropna()
    g = d.groupby(id_col)[val]
    k = g.size()
    d = d[d[id_col].isin(k[k >= 2].index)]
    g = d.groupby(id_col)[val]
    n, N = g.ngroups, len(d)
    k0 = (N - (g.size() ** 2).sum() / N) / (n - 1)
    grand = d[val].mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (n - 1)
    msw = ((d[val] - g.transform("mean")) ** 2).sum() / (N - n)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def sb(r1: float, k: float) -> float:
    return k * r1 / (1 + (k - 1) * r1)


def reps_for(r1: float, target: float) -> float:
    return target * (1 - r1) / (r1 * (1 - target))


def loco(df: pd.DataFrame, base: list[str], extra: list[str], y: str):
    """Leave-one-class-out OLS. Returns pooled out-of-class R2 for base and base+extra, per-class deltas, bootstrap CI."""
    preds_b, preds_a, ys, yrs = [], [], [], []
    for yr in sorted(df.draft_year.unique()):
        tr, te = df[df.draft_year != yr], df[df.draft_year == yr]
        if len(te) < 5:
            continue
        for cols, store in ((base, preds_b), (base + extra, preds_a)):
            A = np.c_[np.ones(len(tr)), tr[cols].values]
            beta = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
            store.append(np.c_[np.ones(len(te)), te[cols].values] @ beta)
        ys.append(te[y].values)
        yrs.append(np.full(len(te), yr))
    yv, pb, pa, yr = map(np.concatenate, (ys, preds_b, preds_a, yrs))

    def r2(yy, pp):
        return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    rb, ra = r2(yv, pb), r2(yv, pa)
    per = {int(c): float(r2(yv[yr == c], pa[yr == c]) - r2(yv[yr == c], pb[yr == c])) for c in np.unique(yr)}
    n = len(yv)
    bd = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        bd.append(r2(yv[i], pa[i]) - r2(yv[i], pb[i]))
    lo, hi = np.percentile(bd, [2.5, 97.5])
    return dict(r2_base=float(rb), r2_aug=float(ra), delta=float(ra - rb), lo=float(lo), hi=float(hi), n=int(n), per_class=per)


# ----------------------------------------------------------------------------- main
def main() -> None:
    d = insights.load()
    p = d["p"].copy()
    D, G, PK, top40, t40_pairs, gp, n_drill_events = build_curves(p)
    cols_b = [f"c{i}" for i in range(BURST)]
    D["burst"] = D[cols_b].mean(axis=1)
    G["burst"] = G[cols_b].mean(axis=1)
    D["burst_z"] = D.groupby(["grp", "drill", "yr"]).burst.transform(zscore)
    G["wide"] = G.lined_up_position.isin(WIDE_ALIGN).astype(float)
    G["burst_z"] = G.groupby(["grp", "lined_up_position"]).burst.transform(zscore)
    PK["peak_z"] = PK.groupby(["grp", "lined_up_position"]).peak.transform(zscore)

    put("n_players", len(p), "{:d}")
    put("n_game_plays", len(gp), "{:,}")
    put("n_game_players", gp.nfl_id.nunique(), "{:d}")
    put("n_drill_reps", len(D), "{:,}")
    put("n_game_curves", len(G), "{:,}")
    put("game_curve_share", 100 * len(G) / len(gp), "{:.0f}")
    put("burst_window_s", 1.0, "{:.1f}")
    put("onset_yps", ON, "{:.1f}")

    # ---- per-player tables
    dm = D.groupby("nfl_id").agg(drill_z=("burst_z", "mean"), drill_burst=("burst", "mean"), n_drill=("burst", "size"))
    gm = G.groupby("nfl_id").agg(game_z=("burst_z", "mean"), game_burst=("burst", "mean"), n_game=("burst", "size"),
                                 wide=("wide", "mean"), getoff_sample=("player_get_off", "mean"))
    gmr = G[G.random_half].groupby("nfl_id").agg(game_z_rand=("burst_z", "mean"))
    pk = PK[PK.random_half].groupby("nfl_id").agg(peak_z=("peak_z", "mean"))
    pk_top3_all = PK.groupby("nfl_id").peak.apply(lambda v: v.nlargest(3).mean()).rename("peak3_all")
    pk_top3_rand = PK[PK.random_half].groupby("nfl_id").peak.apply(lambda v: v.nlargest(3).mean()).rename("peak3_rand")
    x = (p.merge(dm, on="nfl_id").merge(gm, on="nfl_id").merge(gmr, on="nfl_id", how="left")
         .merge(pk, on="nfl_id", how="left").merge(pk_top3_all, on="nfl_id", how="left")
         .merge(pk_top3_rand, on="nfl_id", how="left").merge(top40[["nfl_id", "top40_z", "top_speed"]], on="nfl_id", how="left"))
    # re-standardize player means within group so both sides of the difference share one scale
    for c in ("drill_z", "game_z", "game_z_rand"):
        x[c] = x.groupby("position_group")[c].transform(zscore)
    x["bt"] = x.game_z - x.drill_z            # Burst Translation
    x["bt_rand"] = x.game_z_rand - x.drill_z
    # outcomes, oriented so higher = better
    for g in GROUPS:
        col, basis, mn, hib, _ = insights.OUTCOME[g]
        m = x.position_group == g
        x.loc[m, "y"] = np.where(x.loc[m, basis] >= mn, x.loc[m, col] * (1 if hib else -1), np.nan)
    x["forty_z"] = x.groupby(["position_group", "draft_year"]).forty.transform(zscore)
    x["forty_z"] = x.groupby("position_group").forty_z.transform(lambda v: v.fillna(0.0))
    x["ngs_z"] = x.groupby(["position_group", "draft_year"]).ngs_athleticism_score.transform(zscore).fillna(0.0)
    x["weight_z"] = x.groupby("position_group").combine_weight.transform(zscore)
    put("n_matched_players", len(x), "{:d}")

    # ---- population curves (fig 1)
    fig, axes = plt.subplots(1, 5, figsize=(FIGW, 3.6), sharey=True)
    tt = np.arange(T) * DT
    pop = {}
    for ax, g in zip(axes, GROUPS):
        dc = D[D.grp == g][[f"c{i}" for i in range(T)]]
        gc = G[G.grp == g][[f"c{i}" for i in range(T)]]
        for c, lab, ls in ((dc, "Combine drill", "--"), (gc, "NFL snaps", "-")):
            ax.plot(tt, c.mean().values, ls, color=GCOL[g], lw=2.2, label=lab)
            ax.fill_between(tt, c.quantile(.25).values, c.quantile(.75).values, color=GCOL[g], alpha=.13)
        ax.axvspan(0, 1.0, color=OI["grey"], alpha=.07)
        ax.set_title(f"{g}: {DRILL_LABEL[g]}\nvs {GAME_TYPE[g]}", fontsize=9.5)
        ax.set_xlabel("s after movement onset")
        pop[g] = (float(dc[cols_b].mean(axis=1).mean()), float(gc[cols_b].mean(axis=1).mean()))
    axes[0].set_ylabel("speed (yd/s)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Average speed profile from first movement: Combine drill (dashed) vs NFL snaps (solid); band = IQR, shaded = first second", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_profiles.png", dpi=150)
    plt.close(fig)
    for g in GROUPS:
        put(f"pop_drill_burst_{g}", pop[g][0])
        put(f"pop_game_burst_{g}", pop[g][1])
        put(f"pop_ratio_{g}", 100 * pop[g][1] / pop[g][0], "{:.0f}")

    # ---- reliability and translation per group (table 1, fig 2)
    rows = []
    rel = {}
    for g in GROUPS:
        sub = x[x.position_group == g]
        Dg, Gg = D[D.grp == g], G[G.grp == g]
        r1_d = icc1(Dg, "nfl_id", "burst_z")
        r1_g = icc1(Gg, "nfl_id", "burst_z")
        k_d = float(Dg.groupby("nfl_id").size().mean())
        k_g = float(Gg.groupby("nfl_id").size().mean())
        rel_d, rel_g = sb(r1_d, k_d), sb(r1_g, k_g)
        rel[g] = dict(r1_d=r1_d, r1_g=r1_g, k_d=k_d, k_g=k_g, rel_d=rel_d, rel_g=rel_g)
        mtc, lo, hi, n = spearman_ci(sub.drill_z.values, sub.game_z.values)
        ceiling = float(np.sqrt(max(rel_d, 0) * max(rel_g, 0)))
        # top speed translation (40 top speed -> mean per-snap peak speed, random half)
        ts, tlo, thi, tn = spearman_ci(sub.top40_z.values, sub.peak_z.values)
        sy = sub.dropna(subset=["y"])
        pr = partial_spearman(sy.bt, sy.y, sy[["log_pick", "wide"]])
        pg = partial_spearman(sy.game_z, sy.y, sy[["log_pick", "wide"]])
        pdz = partial_spearman(sy.drill_z, sy.y, sy[["log_pick", "wide"]])
        rows.append(dict(group=g, n=n, n_outcome=pr[2], drill_reps=k_d, game_snaps=k_g, rel_drill=rel_d, rel_game=rel_g,
                         mtc=mtc, mtc_lo=lo, mtc_hi=hi, ceiling=ceiling, top_mtc=ts, top_lo=tlo, top_hi=thi, top_n=tn,
                         bt_y=pr[0], bt_y_lo=pr[3], bt_y_hi=pr[4], bt_y_p=pr[1], game_y=pg[0], game_y_lo=pg[3], game_y_hi=pg[4],
                         game_y_p=pg[1], drill_y=pdz[0], drill_y_p=pdz[1]))
        for kk, v in rows[-1].items():
            if kk == "group":
                continue
            fmt = "{:d}" if kk in ("n", "n_outcome", "top_n") else ("{:.1f}" if kk in ("drill_reps", "game_snaps") else
                                                                  ("{:.3f}" if kk.endswith("_p") else "{:.2f}"))
            put(f"{kk}_{g}", v, fmt)
        put(f"r1_drill_{g}", r1_d)
        put(f"r1_game_{g}", r1_g)
    T1 = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIGW, 4.6))
    ypos = np.arange(len(GROUPS))[::-1]
    for yv, r in zip(ypos, T1.itertuples()):
        ax.plot([r.mtc_lo, r.mtc_hi], [yv + .15] * 2, color=GCOL[r.group], lw=2.5)
        ax.plot(r.mtc, yv + .15, "o", color=GCOL[r.group], ms=9, label="First-second burst: drill vs game" if r.group == "DL" else None)
        ax.plot([r.top_lo, r.top_hi], [yv - .15] * 2, color=OI["grey"], lw=2.5)
        ax.plot(r.top_mtc, yv - .15, "s", color=OI["grey"], ms=8, label="Top speed: 40-yd dash vs game peak" if r.group == "DL" else None)
        ax.plot(r.ceiling, yv + .15, "|", color=OI["black"], ms=18, mew=2.5, label="Burst ceiling set by measurement noise" if r.group == "DL" else None)
        ax.text(1.02, yv, f"n={r.n}", va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=OI["black"], lw=.8)
    ax.set_yticks(ypos, [f"{g}" for g in GROUPS])
    ax.set_xlim(-0.75, 1.0)
    ax.set_xlabel("Spearman rho across players (95% bootstrap CI)")
    ax.set_title("Movement Translation Coefficient: how much of a player's Combine rank reappears on NFL snaps", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_translation.png", dpi=150)
    plt.close(fig)

    # ---- reps needed (fig 3) for focus group
    rf = rel[FOCUS]
    ks = np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(FIGW, 4.2))
    ax.plot(ks, [sb(rf["r1_d"], k) for k in ks], "--", color=GCOL[FOCUS], lw=2.2, label=f"Combine pass-rush drill reps (single-rep reliability {rf['r1_d']:.2f})")
    ax.plot(ks, [sb(rf["r1_g"], k) for k in ks], "-", color=OI["blue"], lw=2.2, label=f"NFL pass-rush snaps (single-snap reliability {rf['r1_g']:.2f})")
    ax.axhline(0.7, color=OI["grey"], lw=1, ls=":")
    ax.plot([rf["k_d"]], [rf["rel_d"]], "o", color=GCOL[FOCUS], ms=9)
    ax.plot([rf["k_g"]], [rf["rel_g"]], "o", color=OI["blue"], ms=9)
    ax.annotate(f"what the Combine gives: {rf['k_d']:.1f} reps", (rf["k_d"], rf["rel_d"]), xytext=(6, -28), textcoords="offset points", fontsize=9)
    ax.annotate(f"our NFL sample: {rf['k_g']:.1f} snaps", (rf["k_g"], rf["rel_g"]), xytext=(6, -22), textcoords="offset points", fontsize=9)
    ax.set_xlabel("number of reps averaged")
    ax.set_ylabel("reliability of first-second burst")
    ax.set_ylim(0, 1)
    ax.set_title("DL first-second burst reliability vs reps averaged (dotted line = 0.7): the Combine stops at about two reps", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_reliability.png", dpi=150)
    plt.close(fig)
    put("reps07_drill_DL", reps_for(rf["r1_d"], 0.7), "{:.0f}")
    put("reps07_game_DL", reps_for(rf["r1_g"], 0.7), "{:.1f}")

    # ---- focus group scatter (fig 4) and named players
    sx = x[x.position_group == FOCUS].copy()
    sx["press"] = sx.pressure_rate.where(sx.pass_rush_snaps >= 100)
    q = sx.dropna(subset=["press"]).sort_values("bt")
    named = pd.concat([q.tail(3), q.head(3)])
    fig, ax = plt.subplots(figsize=(FIGW, 5.6))
    nn_ = sx[sx.press.isna()]
    ax.scatter(nn_.drill_z, nn_.game_z, s=22, color=OI["grey"], alpha=.45, label="< 100 NFL rush snaps")
    sc = ax.scatter(q.drill_z, q.game_z, c=100 * q.press, cmap="cividis", s=60, edgecolor="k", lw=.4)
    cb = fig.colorbar(sc, ax=ax, pad=.01)
    cb.set_label("career pressure rate (%)")
    lim = 3.4
    ax.plot([-lim, lim], [-lim, lim], color=OI["grey"], lw=1, ls=":")
    ax.text(2.0, 2.3, "game = drill", color=OI["grey"], fontsize=9, rotation=33)
    for j, r in enumerate(named.itertuples()):
        ax.annotate(f"{r.display_name} (BT {r.bt:+.1f})", (r.drill_z, r.game_z), xytext=(-150 if r.drill_z > 0.8 else 8, 12 if j % 2 else -16),
                    textcoords="offset points", fontsize=8.5, arrowprops=dict(arrowstyle="-", color=OI["grey"], lw=.6))
    ax.set_xlabel("Combine pass-rush drill first-second burst (z within class)")
    ax.set_ylabel("NFL pass-rush first-second burst (z within alignment)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    mtc = T1.set_index("group").loc[FOCUS]
    ax.set_title(f"DL: Combine burst barely predicts NFL burst (rho = {mtc.mtc:.2f}, n = {int(mtc.n)}); "
                 "Burst Translation (BT) = vertical distance from the diagonal", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_dl_scatter.png", dpi=150)
    plt.close(fig)
    named_out = []
    for i, r in enumerate(named.itertuples()):
        named_out.append(dict(name=r.display_name, pick=int(r.pick), drill_z=round(r.drill_z, 2), game_z=round(r.game_z, 2),
                              bt=round(r.bt, 2), press=round(100 * r.press, 1), snaps=int(r.pass_rush_snaps)))
        tag = f"named{i}"
        put(f"{tag}_bt", r.bt, "{:+.1f}")
        put(f"{tag}_press", 100 * r.press, "{:.1f}")
        put(f"{tag}_pick", int(r.pick), "{:d}")
        put(f"{tag}_drill_z", r.drill_z, "{:+.2f}")
        put(f"{tag}_game_z", r.game_z, "{:+.2f}")
    top_mean = float(q.tail(int(len(q) / 4)).press.mean() * 100)
    bot_mean = float(q.head(int(len(q) / 4)).press.mean() * 100)
    put("dl_bt_q4_press", top_mean, "{:.1f}")
    put("dl_bt_q1_press", bot_mean, "{:.1f}")
    put("dl_n_press", len(q), "{:d}")

    # ---- named player curves (fig 5)
    pick4 = [named.iloc[0], named.iloc[1], named.iloc[3], named.iloc[4]]
    fig, axes = plt.subplots(1, 4, figsize=(FIGW, 3.5), sharey=True)
    for ax, r in zip(axes, pick4):
        dc = D[(D.nfl_id == r.nfl_id)][[f"c{i}" for i in range(T)]].mean()
        gc = G[(G.nfl_id == r.nfl_id)][[f"c{i}" for i in range(T)]]
        for _, row in gc.iterrows():
            ax.plot(tt, row.values, color=OI["blue"], alpha=.18, lw=1)
        ax.plot(tt, gc.mean().values, color=OI["blue"], lw=2.4, label="NFL rush snaps (mean)")
        ax.plot(tt, dc.values, "--", color=GCOL[FOCUS], lw=2.4, label="Combine pass-rush drill")
        ax.plot(tt, D[D.grp == FOCUS][[f"c{i}" for i in range(T)]].mean().values, ":", color=OI["grey"], lw=1.5, label="DL drill average")
        ax.set_title(f"{r.display_name}\nBT {r.bt:+.1f}, pressure {100 * r.press:.1f}%", fontsize=9.5)
        ax.set_xlabel("s after onset")
    axes[0].set_ylabel("speed (yd/s)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_named.png", dpi=150)
    plt.close(fig)

    # ---- out-of-class validation (fig 6)
    base = ["log_pick", "wide", "weight_z", "forty_z", "ngs_z"]
    lo_rows = []
    for g in GROUPS:
        sub = x[(x.position_group == g)].dropna(subset=["y", "bt"] + base)
        for lab, extra in (("Burst Translation", ["bt"]), ("NFL burst only", ["game_z"]), ("Combine burst only", ["drill_z"])):
            res = loco(sub, base, extra, "y")
            lo_rows.append(dict(group=g, model=lab, **{k: v for k, v in res.items() if k != "per_class"}, per_class=res["per_class"]))
    LO = pd.DataFrame(lo_rows)
    for r in LO.itertuples():
        key = {"Burst Translation": "bt", "NFL burst only": "game", "Combine burst only": "drill"}[r.model]
        put(f"loco_{key}_delta_{r.group}", r.delta, "{:.3f}")
        put(f"loco_{key}_lo_{r.group}", r.lo, "{:.3f}")
        put(f"loco_{key}_hi_{r.group}", r.hi, "{:.3f}")
        put(f"loco_base_r2_{r.group}", r.r2_base, "{:.2f}")
        put(f"loco_n_{r.group}", r.n, "{:d}")
        put(f"loco_{key}_classes_pos_{r.group}", sum(v > 0 for v in r.per_class.values()), "{:d}")
    # DL: does NFL burst add beyond NGS get-off?
    sub = x[(x.position_group == FOCUS)].dropna(subset=["y", "bt", "avg_get_off"] + base)
    go = loco(sub, base + ["avg_get_off"], ["game_z"], "y")
    go_bt = loco(sub, base + ["avg_get_off"], ["bt"], "y")
    go_base = loco(sub, base, ["avg_get_off"], "y")
    put("dl_getoff_base_r2", go["r2_base"], "{:.2f}")
    put("dl_getoff_add_delta", go_base["delta"], "{:.3f}")
    put("dl_game_over_getoff_delta", go["delta"], "{:.3f}")
    put("dl_game_over_getoff_lo", go["lo"], "{:.3f}")
    put("dl_game_over_getoff_hi", go["hi"], "{:.3f}")
    put("dl_bt_over_getoff_delta", go_bt["delta"], "{:.3f}")
    put("dl_getoff_n", go["n"], "{:d}")
    put("dl_rho_game_getoff", stats.spearmanr(sx.game_z, sx.avg_get_off, nan_policy="omit")[0])

    fig, ax = plt.subplots(figsize=(FIGW, 4.6))
    models = ["Combine burst only", "Burst Translation", "NFL burst only"]
    PG = [g for g in GROUPS if g != "TE"]   # TE (n=21) is too small for a three-fold class split
    mcol = {"Combine burst only": GCOL[FOCUS], "Burst Translation": OI["black"], "NFL burst only": OI["blue"]}
    for j, mname in enumerate(models):
        sub = LO[LO.model == mname].set_index("group").loc[PG]
        xs = np.arange(len(PG)) + (j - 1) * 0.25
        ax.errorbar(xs, sub.delta, yerr=[sub.delta - sub.lo, sub.hi - sub.delta], fmt="o", color=mcol[mname], ms=8, capsize=3, lw=2, label=f"+ {mname}")
    ax.axhline(0, color=OI["black"], lw=.8)
    ax.set_xticks(np.arange(len(PG)), [f"{g}\n{OUTCOME_LABEL[g]}\nn={int(LO[(LO.group == g)].n.iloc[0])}" for g in PG], fontsize=8.5)
    ax.set_ylabel("change in out-of-class R2\nvs pick + alignment + weight + 40 + NGS")
    ax.set_title("Train on two draft classes, predict the third: no burst variant reliably beats the no-tracking baseline", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_out_of_class.png", dpi=150)
    plt.close(fig)

    # ---- sensitivity: random half of plays only (unbiased by the ETL's top-|EPA| half)
    for g in GROUPS:
        sub = x[x.position_group == g]
        r_, lo_, hi_, n_ = spearman_ci(sub.drill_z.values, sub.game_z_rand.values)
        put(f"mtc_rand_{g}", r_)
        sy = sub.dropna(subset=["y", "bt_rand"])
        pr = partial_spearman(sy.bt_rand, sy.y, sy[["log_pick", "wide"]])
        put(f"bt_rand_y_{g}", pr[0])
        put(f"bt_rand_y_p_{g}", pr[1], "{:.3f}")
    # selection artifact: best-effort speed (top-3 per-snap peaks) / Combine top speed vs snaps per season
    art = []
    for g in GROUPS:
        sub = x[x.position_group == g]
        a1 = partial_spearman(sub.peak3_all / sub.top_speed, sub.snaps_per_season, sub[["log_pick", "wide"]])
        a2 = partial_spearman(sub.peak3_rand / sub.top_speed, sub.snaps_per_season, sub[["log_pick", "wide"]])
        art.append(dict(group=g, all=a1[0], all_p=a1[1], rand=a2[0], rand_p=a2[1], n=a1[2]))
        put(f"art_all_{g}", a1[0])
        put(f"art_all_p_{g}", a1[1], "{:.3f}")
        put(f"art_rand_{g}", a2[0])
        put(f"art_rand_p_{g}", a2[1], "{:.3f}")
        put(f"art_n_{g}", a1[2], "{:d}")
    # 40 top-speed repeatability for context
    t40 = np.vstack(t40_pairs.values)
    put("top40_retest", stats.pearsonr(t40[:, 0], t40[:, 1])[0])
    put("top40_retest_n", len(t40), "{:d}")
    # population-level DL contact plateau: time of peak in mean game curve
    gcm = G[G.grp == FOCUS][[f"c{i}" for i in range(T)]].mean().values
    put("dl_game_peak_t", tt[int(np.argmax(gcm))], "{:.1f}")
    put("dl_game_peak_v", gcm.max())
    dcm = D[D.grp == FOCUS][[f"c{i}" for i in range(T)]].mean().values
    put("dl_drill_v15", dcm[-1])
    put("dl_diverge_t", tt[int(np.argmax(gcm < 0.95 * dcm))], "{:.1f}")

    T1.round(3).to_json(HERE / "tables.json", orient="records", indent=1)
    out = dict(NUM)
    out["_abs_values"] = sorted(ABS)
    out["_named"] = named_out
    out["_loco"] = LO.drop(columns=["per_class"]).round(4).to_dict("records")
    out["_artifact"] = pd.DataFrame(art).round(3).to_dict("records")
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1, default=str))
    log(T1.round(2).to_string())
    log(LO.drop(columns=["per_class"]).round(3).to_string())
    log(json.dumps(named_out, indent=1))
    log(pd.DataFrame(art).round(3).to_string())
    log(json.dumps({k: v for k, v in NUM.items() if k.startswith(("dl_", "reps", "pop_", "top40", "mtc_rand", "bt_rand", "game_curve", "n_"))}, indent=1))


if __name__ == "__main__":
    main()
