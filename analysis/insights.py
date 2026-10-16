"""First batch of data insights: what the draft pays for, what the tracking adds, and a v0 value model.

Reads data/processed/*.parquet, writes data/processed/insights.json (chart-ready) and
data/processed/player_value.parquet (expected vs actual per player).

Usage: uv run python analysis/insights.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdb27.drills import DRILL_BY_NAME  # noqa: E402

PROC = ROOT / "data" / "processed"
GROUPS = ["WR", "TE", "OL", "DL", "DB"]
UDFA_PICK = 262  # one slot after the last pick of a 7-round draft
MEASURABLES = ["combine_height", "combine_weight", "arm_length", "hand_size", "wing_span", "forty", "ten_yd_split",
               "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps",
               "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
LABEL = {"combine_height": "Height", "combine_weight": "Weight", "arm_length": "Arm length", "hand_size": "Hand size",
         "wing_span": "Wingspan", "forty": "40-yard", "ten_yd_split": "10-yd split", "vertical": "Vertical", "broad_jump": "Broad jump",
         "three_cone": "3-cone", "short_shuttle": "Shuttle", "bench_reps": "Bench", "ngs_athleticism_score": "NGS athleticism",
         "ngs_college_production_score": "NGS production", "ngs_final_score": "NGS final", "top_speed_40": "Tracked top speed",
         "peak_accel_40": "Tracked peak accel (40)", "drill_pct": "Avg drill percentile", "log_pick": "Draft pick (log)"}
# per-group NFL outcome: (column in career-aggregated stats, basis column, min basis, higher_is_better, label)
OUTCOME = {
    "WR": ("yards_per_route", "routes", 150, True, "Yards per route run"),
    "TE": ("yards_per_route", "routes", 100, True, "Yards per route run"),
    "OL": ("pressure_rate_allowed", "pass_blocks", 150, False, "Pressure rate allowed"),
    "DL": ("pressure_rate", "pass_rush_snaps", 100, True, "Pressure rate"),
    "DB": ("tackles_per_100", "plays", 150, True, "Solo tackles per 100 snaps"),
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    if m.sum() < 12:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def load() -> dict[str, pd.DataFrame]:
    d = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in
         ["players", "combine_results", "combine_events", "player_season_stats", "player_career"]}
    p = d["players"].merge(d["combine_results"].drop(columns=["draft_year", "combine_position", "combine_height", "combine_weight"]), on="nfl_id")
    p = p.merge(d["player_career"], on="nfl_id", how="left")
    p["pick"] = p["draft_overall_pick"].fillna(UDFA_PICK)
    p["log_pick"] = np.log(p["pick"])
    p["seasons_available"] = 2026 - p["draft_year"]
    p["snaps_per_season"] = p["career_scrimmage_snaps"] / p["seasons_available"]
    p["starts_per_season"] = p["career_games_started"] / p["seasons_available"]
    ev = d["combine_events"]
    f40 = ev[ev.drill_type == "FORTY_YARD_DASH"].groupby("nfl_id").agg(top_speed_40=("top_speed", "max"), peak_accel_40=("peak_accel", "max"), split_40_tracked=("split_40yd_s", "min"), split_10_tracked=("split_10yd_s", "min"))
    dp = ev[ev.drill_group.isin(GROUPS)].groupby("nfl_id").agg(drill_pct=("key_metric_pct", "mean"), n_pos_drills=("event_id", "count"))
    p = p.merge(f40, on="nfl_id", how="left").merge(dp, on="nfl_id", how="left")
    # career-aggregated NFL outcome (regular season, weighted by basis)
    s = d["player_season_stats"]
    agg = s.groupby("nfl_id").agg(routes=("routes", "sum"), rec_yards=("rec_yards", "sum"), targets=("targets", "sum"),
                                  pass_blocks=("pass_blocks", "sum"), pressures_allowed=("pressures_allowed", "sum"),
                                  pass_rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                                  plays=("plays", "sum"), tackles=("tackles", "sum"), sep_sum=("avg_separation", lambda v: np.nan),
                                  seasons_played=("season", "count"))
    sep = s.dropna(subset=["avg_separation"]).assign(w=lambda x: x.routes * x.avg_separation).groupby("nfl_id").agg(w=("w", "sum"), r=("routes", "sum"))
    agg["avg_separation"] = (sep.w / sep.r).reindex(agg.index)
    getoff = s.dropna(subset=["avg_get_off"]).assign(w=lambda x: x.pass_rush_snaps * x.avg_get_off).groupby("nfl_id").agg(w=("w", "sum"), r=("pass_rush_snaps", "sum"))
    agg["avg_get_off"] = (getoff.w / getoff.r).reindex(agg.index)
    agg["yards_per_route"] = agg.rec_yards / agg.routes
    agg["pressure_rate_allowed"] = agg.pressures_allowed / agg.pass_blocks
    agg["pressure_rate"] = agg.pressures / agg.pass_rush_snaps
    agg["tackles_per_100"] = 100 * agg.tackles / agg.plays
    p = p.merge(agg.drop(columns=["sep_sum"]), on="nfl_id", how="left")
    d["p"] = p
    return d


# ----------------------------------------------------------------------------- insights
def insight_draft_pays(p: pd.DataFrame) -> dict:
    rows = []
    for g in GROUPS:
        sub = p[p.position_group == g]
        for m in MEASURABLES:
            r, pv, n = spearman(sub[m], sub["pick"])
            # sign so that positive = "better value → earlier pick"
            better_low = m in ("forty", "ten_yd_split", "three_cone", "short_shuttle")
            rows.append({"group": g, "metric": LABEL[m], "rho": round((-r if not better_low else r), 3) if not np.isnan(r) else None, "p": round(pv, 4) if not np.isnan(pv) else None, "n": n})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="metric", columns="group", values="rho").reindex([LABEL[m] for m in MEASURABLES])
    strongest = df.dropna().sort_values("rho", ascending=False).head(5)
    summary = ("Spearman rank correlation between each Combine number and draft slot (UDFA = pick 262), signed so that positive means 'better number, earlier pick'. "
               + "Strongest links: " + "; ".join(f"{r.metric} for {r.group} (ρ={r.rho:.2f}, n={r.n})" for r in strongest.itertuples()) + ".")
    return {"id": "draft_pays", "title": "What the draft market pays for", "section": "market",
            "summary": summary,
            "method": "Spearman ρ per position group; p-values two-sided; n = players with both values.",
            "chart": {"type": "heatmap", "rows": list(piv.index), "cols": list(piv.columns), "values": piv.round(2).where(piv.notna(), None).values.tolist(), "scale": [-0.6, 0.6], "fmt": "rho"},
            "table": df.dropna().sort_values("rho", ascending=False).head(12).to_dict("records")}


def insight_tracked_vs_official(p: pd.DataFrame) -> dict:
    m = p.dropna(subset=["split_40_tracked", "forty"])
    r = stats.pearsonr(m.split_40_tracked, m.forty)
    offset = (m.split_40_tracked - m.forty)
    m10 = p.dropna(subset=["split_10_tracked", "ten_yd_split"])
    r10 = stats.pearsonr(m10.split_10_tracked, m10.ten_yd_split)
    ms = p.dropna(subset=["top_speed_40", "forty"])
    rs = stats.pearsonr(ms.top_speed_40, ms.forty)
    pts = m.sample(min(len(m), 420), random_state=1)
    return {"id": "tracked_vs_official", "title": "The sensor sees the finish, not the start", "section": "data",
            "summary": (f"Elapsed time from the first tracked frame to 40 yards correlates r={r[0]:.2f} with the official laser time (n={len(m)}), but is {offset.mean():.2f} s longer on average "
                        f"(sd {offset.std():.2f}): the clip includes the stance and the first rock before the timer starts on forward movement, and that pre-start window varies by player. The 10-yard split agrees far worse (r={r10[0]:.2f}, n={len(m10)}) because it sits entirely inside that noisy start, while tracked top speed alone predicts the official 40 at |r|={abs(rs[0]):.2f} (n={len(ms)}). "
                        "Use the tracking for the shape of the run (where top speed is reached, how it decays), never as a replacement for the stopwatch, and treat the first 10 yards with suspicion."),
            "method": "Best attempt per player; 40-yard elapsed from x displacement; Pearson r.",
            "chart": {"type": "scatter", "x": "Tracked 40 (s, from first frame)", "y": "Official 40 (s)",
                      "points": [{"x": round(a, 2), "y": round(b, 2), "g": g, "label": n} for a, b, g, n in zip(pts.split_40_tracked, pts.forty, pts.position_group, pts.display_name)],
                      "identity": True},
            "table": [{"comparison": "40-yd elapsed vs official 40", "r": round(r[0], 3), "n": len(m)}, {"comparison": "10-yd elapsed vs official 10 split", "r": round(r10[0], 3), "n": len(m10)}, {"comparison": "Tracked top speed vs official 40", "r": round(rs[0], 3), "n": len(ms)}]}


def insight_year_effect(ev: pd.DataFrame) -> dict:
    rows = []
    for name, g in ev[ev.drill_group.isin(GROUPS + ["SPEED", "AGILITY"])].groupby("drill_name"):
        d = DRILL_BY_NAME.get(name)
        if not d:
            continue
        med = g.groupby("draft_year")[d["key_metric"]].median()
        if len(med) < 2:
            continue
        sd = g[d["key_metric"]].std()
        spread = (med.max() - med.min()) / sd if sd and sd > 0 else np.nan
        rows.append({"drill": d["label"], "group": d["drill_group"], "metric": d["key_metric"], **{str(y): round(float(v), 2) for y, v in med.items()}, "spread_sd": round(float(spread), 2), "n": int(len(g))})
    df = pd.DataFrame(rows).sort_values("spread_sd", ascending=False)
    worst = df.head(6)
    return {"id": "year_effect", "title": "Drills are not comparable across Combine years without adjustment", "section": "data",
            "summary": ("Median of each drill's key metric by Combine year. The spread is expressed in within-drill standard deviations; anything above ~0.5 means a player's percentile depends more on which year he attended than on how he moved. "
                        + "Largest gaps: " + "; ".join(f"{r.drill} ({r.spread_sd:.1f} sd)" for r in worst.itertuples()) + ". Every cross-class comparison in the analysis is therefore made on year-standardized values."),
            "method": "Median key metric per draft year; spread = (max − min) / pooled sd.",
            "chart": {"type": "table"},
            "table": df.head(20).to_dict("records")}


def insight_reliability(ev: pd.DataFrame) -> dict:
    rows = []
    for name, g in ev.groupby("drill_name"):
        d = DRILL_BY_NAME.get(name)
        if not d:
            continue
        two = g[g.attempt_no.isin([1, 2])].pivot_table(index="nfl_id", columns="attempt_no", values=d["key_metric"])
        if 1 not in two.columns or 2 not in two.columns:
            continue
        two = two.dropna()
        if len(two) < 15:
            continue
        r = stats.spearmanr(two[1], two[2])[0]
        rows.append({"drill": d["label"], "group": d["drill_group"], "metric": d["key_metric"], "rho_attempts": round(float(r), 2), "players_with_2": int(len(two))})
    df = pd.DataFrame(rows).sort_values("rho_attempts", ascending=False)
    return {"id": "reliability", "title": "Most position drills are single-attempt, and the repeats barely agree", "section": "data",
            "summary": (f"Only {len(df)} drills have at least 15 players with two attempts. Where they exist, the rank correlation between attempt 1 and attempt 2 of the key metric ranges from "
                        f"{df.rho_attempts.min():.2f} to {df.rho_attempts.max():.2f}. The 40 and the timed agility drills repeat well; reaction-based position drills do not, so any single-attempt drill metric carries large measurement error and must be shrunk toward the position mean before it is used to rank prospects."),
            "method": "Spearman ρ between attempt 1 and attempt 2 of the key metric; drills with ≥15 such players.",
            "chart": {"type": "bar", "x": "drill", "y": "rho_attempts", "label": "ρ attempt 1 vs 2", "domain": [-0.2, 1]},
            "table": df.to_dict("records")}


def insight_predicts_nfl(p: pd.DataFrame) -> dict:
    preds = ["log_pick", "forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_weight", "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score", "top_speed_40", "peak_accel_40", "drill_pct"]
    rows, cols, vals, ns = [], [], [], []
    for g in GROUPS:
        out, basis, mn, hib, lab = OUTCOME[g]
        sub = p[(p.position_group == g) & (p[basis] >= mn)]
        for m in preds:
            r, pv, n = spearman(sub[m], sub[out])
            if not hib:
                r = -r
            if m in ("forty", "ten_yd_split", "three_cone", "short_shuttle", "log_pick"):
                r = -r  # lower is better → sign flip so positive = good number, good outcome
            rows.append({"group": g, "outcome": lab, "predictor": LABEL[m], "rho": round(r, 3) if not np.isnan(r) else None, "p": round(pv, 4) if not np.isnan(pv) else None, "n": n})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="predictor", columns="group", values="rho").reindex([LABEL[m] for m in preds])
    sig = df[(df.p < 0.05)].sort_values("rho", key=abs, ascending=False)
    opp = p[p.position_group.isin(GROUPS)].copy()
    r_opp = {g: spearman(opp[opp.position_group == g]["log_pick"], opp[opp.position_group == g]["snaps_per_season"])[0] for g in GROUPS}
    return {"id": "predicts_nfl", "title": "What actually correlates with NFL production", "section": "nfl",
            "summary": (f"Rank correlation between pre-draft numbers and a per-position production rate (WR/TE yards per route, OL pressure rate allowed, DL pressure rate, DB solo tackles per 100 snaps), signed so positive = better number, better outcome. "
                        f"Draft slot is the strongest signal of opportunity everywhere (ρ with snaps per season from {min(r_opp.values()):.2f} to {max(r_opp.values()):.2f}), but of production only in some groups. "
                        + ("Statistically clear links (p<0.05): " + "; ".join(f"{r.predictor}→{r.outcome} for {r.group} (ρ={r.rho:.2f}, n={r.n})" for r in sig.head(8).itertuples()) + "." if len(sig) else "No predictor reaches p<0.05.")),
            "method": "Spearman ρ; players above a minimum sample (WR 150 routes, OL 150 pass blocks, DL 100 rush snaps, DB 150 plays). Production is aggregated over all regular seasons played.",
            "chart": {"type": "heatmap", "rows": list(piv.index), "cols": list(piv.columns), "values": piv.round(2).where(piv.notna(), None).values.tolist(), "scale": [-0.5, 0.5], "fmt": "rho"},
            "table": sig.head(15).to_dict("records")}


def value_model(p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Expected opportunity (snaps/season) and expected production given pick + Combine, with CV R² by feature set."""
    feats_pick = ["log_pick"]
    feats_combine = feats_pick + ["forty", "ten_yd_split", "vertical", "broad_jump", "combine_weight", "combine_height", "arm_length", "ngs_athleticism_score", "ngs_college_production_score"]
    feats_track = feats_combine + ["top_speed_40", "peak_accel_40", "drill_pct"]
    sets = {"pick only": feats_pick, "+ Combine numbers": feats_combine, "+ tracking features": feats_track}
    out_rows, r2_rows = [], []
    rng = KFold(5, shuffle=True, random_state=11)
    for g in GROUPS:
        sub = p[(p.position_group == g) & (p.draft_year <= 2025)].copy()
        targets = {"opportunity": ("snaps_per_season", sub.index)}
        out, basis, mn, hib, lab = OUTCOME[g]
        targets["production"] = (out, sub.index[sub[basis] >= mn])
        for tname, (ycol, idx) in targets.items():
            d = sub.loc[idx].dropna(subset=[ycol])
            if len(d) < 25:
                continue
            y = d[ycol].values
            preds_by_set = {}
            for sname, fs in sets.items():
                X = d[fs].copy()
                X = X.fillna(X.median())
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
                pred = cross_val_predict(model, X, y, cv=rng)
                preds_by_set[sname] = pred
                r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                r2_rows.append({"group": g, "target": tname, "features": sname, "cv_r2": round(float(r2), 3), "n": int(len(d))})
            pred = preds_by_set["+ Combine numbers"]
            resid = y - pred
            if not hib and tname == "production":
                resid = -resid  # for pressure-rate-allowed lower is better
            z = (resid - resid.mean()) / resid.std() if resid.std() > 0 else resid * 0
            for i, (nid, yy, pp_, rr, zz) in enumerate(zip(d.nfl_id, y, pred, resid, z)):
                out_rows.append({"nfl_id": int(nid), "position_group": g, "target": tname, "label": "Snaps per season" if tname == "opportunity" else lab,
                                 "actual": round(float(yy), 4), "expected": round(float(pp_), 4), "residual": round(float(rr), 4), "z": round(float(zz), 2), "n_group": int(len(d))})
    pv = pd.DataFrame(out_rows)
    pv["pct_in_group"] = pv.groupby(["position_group", "target"])["z"].rank(pct=True).mul(100).round(1)
    r2 = pd.DataFrame(r2_rows)
    piv = r2.pivot_table(index=["target", "group"], columns="features", values="cv_r2").reindex(columns=list(sets))
    gain = (piv["+ Combine numbers"] - piv["pick only"])
    gain_t = (piv["+ tracking features"] - piv["+ Combine numbers"])
    ins = {"id": "value_model", "title": "Expected vs actual: who beat his draft slot", "section": "value",
           "summary": (f"Ridge regression (5-fold CV) of two outcomes on draft slot alone, then adding Combine numbers, then adding tracking features. Opportunity (snaps per season) is well explained by the pick (CV R² {piv.loc['opportunity', 'pick only'].min():.2f}–{piv.loc['opportunity', 'pick only'].max():.2f}); "
                       f"Combine numbers add between {gain.min():+.2f} and {gain.max():+.2f} R², and tracking features add {gain_t.min():+.2f} to {gain_t.max():+.2f}. "
                       "The residual of the pick + Combine model is the player's 'value over draft slot': positive means more snaps or better production than a prospect with his pick and his numbers normally gets. The tables below list the largest residuals per group; the same number is shown on every player card."),
           "method": "RidgeCV on standardized features, median-imputed; CV R² = 1 − SSE/SST on out-of-fold predictions. Production targets use the same thresholds as the correlation table.",
           "chart": {"type": "table"},
           "table": r2.to_dict("records")}
    return pv, ins


def insight_value_lists(pv: pd.DataFrame, p: pd.DataFrame) -> dict:
    names = p.set_index("nfl_id")[["display_name", "nfl_position", "draft_year", "draft_round", "draft_overall_pick"]]
    rows = []
    for (g, t), d in pv.groupby(["position_group", "target"]):
        d = d.join(names, on="nfl_id")
        for kind, dd in (("over", d.nlargest(5, "z")), ("under", d.nsmallest(5, "z"))):
            for r in dd.itertuples():
                rows.append({"group": g, "target": t, "kind": kind, "nfl_id": r.nfl_id, "player": r.display_name, "pos": r.nfl_position, "class": int(r.draft_year),
                             "pick": "UDFA" if pd.isna(r.draft_overall_pick) else f"R{int(r.draft_round)} #{int(r.draft_overall_pick)}",
                             "label": r.label, "actual": r.actual, "expected": r.expected, "z": r.z})
    return {"id": "value_lists", "title": "Largest over- and under-performers vs expectation", "section": "value",
            "summary": "Top five and bottom five residuals per group for opportunity (snaps per season) and for production. Positive z = did more than the model expected from pick and Combine numbers. Early picks dominate the 'under' lists by construction: the model expects a lot of them.",
            "method": "Residual z-score within group from the pick + Combine ridge model.",
            "chart": {"type": "table"}, "table": rows}


def insight_pass_rush(p: pd.DataFrame, ev: pd.DataFrame) -> dict:
    pr = ev[ev.drill_name == "PASS_RUSH_DRILL"].groupby("nfl_id").agg(pr_peak_accel=("peak_accel", "max"), pr_time_to_5=("time_to_5yps_s", "min"), pr_top=("top_speed", "max"))
    hoop = ev[ev.drill_name == "RUN_THE_HOOP_DRILL"].groupby("nfl_id").agg(hoop_speed=("mean_speed", "max"), hoop_dur=("duration_s", "min"))
    d = p[(p.position_group == "DL") & (p.pass_rush_snaps >= 100)].merge(pr, on="nfl_id", how="left").merge(hoop, on="nfl_id", how="left")
    rows = []
    for m, lab in [("pr_peak_accel", "Pass rush drill: peak accel"), ("pr_time_to_5", "Pass rush drill: time to 5 yd/s"), ("pr_top", "Pass rush drill: top speed"), ("hoop_speed", "Run the hoop: avg speed"), ("hoop_dur", "Run the hoop: duration"), ("ten_yd_split", "Official 10-yd split"), ("forty", "Official 40"), ("combine_weight", "Weight"), ("log_pick", "Draft pick (log)")]:
        for out, olab in [("avg_get_off", "Avg get-off (s)"), ("pressure_rate", "Pressure rate"), ("snaps_per_season", "Snaps per season")]:
            r, pv_, n = spearman(d[m], d[out])
            rows.append({"predictor": lab, "outcome": olab, "rho": round(r, 3) if not np.isnan(r) else None, "p": round(pv_, 4) if not np.isnan(pv_) else None, "n": n})
    df = pd.DataFrame(rows)
    piv = df.pivot(index="predictor", columns="outcome", values="rho")
    pts = d.dropna(subset=["pr_peak_accel", "avg_get_off"])
    r = spearman(pts.pr_peak_accel, pts.avg_get_off)
    return {"id": "pass_rush", "title": "Edge/DL: does the pass-rush drill get-off show up in games?", "section": "nfl",
            "summary": (f"Among {len(d)} DL/EDGE with 100+ pass-rush snaps, peak acceleration in the Combine pass-rush drill correlates ρ={r[0]:.2f} (p={r[1]:.3f}, n={r[2]}) with in-game get-off time (negative = quicker off the ball). "
                        "The heatmap compares every drill-derived number against get-off, pressure rate and opportunity; the official 10-yard split is the benchmark to beat."),
            "method": "Spearman ρ; DL/EDGE with ≥100 regular-season pass-rush snaps; drill values are the best attempt.",
            "chart": {"type": "heatmap", "rows": list(piv.index), "cols": list(piv.columns), "values": piv.round(2).where(piv.notna(), None).values.tolist(), "scale": [-0.5, 0.5], "fmt": "rho"},
            "table": [{"x": round(a, 2), "y": round(b, 3), "label": n} for a, b, n in zip(pts.pr_peak_accel, pts.avg_get_off, pts.display_name)],
            "scatter": {"x": "Pass-rush drill peak accel (yd/s²)", "y": "In-game avg get-off (s)", "points": [{"x": round(a, 2), "y": round(b, 3), "g": "DL", "label": n} for a, b, n in zip(pts.pr_peak_accel, pts.avg_get_off, pts.display_name)]}}


def insight_wr_breaks(p: pd.DataFrame, ev: pd.DataFrame) -> dict:
    routes = ["COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SLANT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "GO_ROUTE_RIGHT"]
    w = p[(p.position_group == "WR") & (p.routes >= 150)]
    rows = []
    for rn in routes:
        g = ev[ev.drill_name == rn].groupby("nfl_id").agg(top=("top_speed", "max"), decel=("break_decel", "max"), smin=("break_speed_min", "max"), out=("break_speed_out", "max"), deg=("sharpest_break_deg", "max"))
        d = w.merge(g, on="nfl_id", how="inner")
        for m, lab in [("top", "top speed"), ("decel", "decel into break"), ("smin", "speed through break"), ("out", "burst out of break"), ("deg", "break angle")]:
            for out, olab in [("avg_separation", "Avg separation"), ("yards_per_route", "Yards per route")]:
                r, pv_, n = spearman(d[m], d[out])
                rows.append({"drill": DRILL_BY_NAME[rn]["label"], "feature": lab, "outcome": olab, "rho": round(r, 3) if not np.isnan(r) else None, "p": round(pv_, 4) if not np.isnan(pv_) else None, "n": n})
    df = pd.DataFrame(rows).dropna()
    best = df.sort_values("rho", key=abs, ascending=False).head(8)
    return {"id": "wr_breaks", "title": "WR: route-drill break mechanics vs in-game separation", "section": "nfl",
            "summary": (f"For wide receivers with 150+ routes (n≈{int(df.n.median())} per drill), rank correlations between break metrics from each route drill and NFL separation / yards per route. "
                        "Largest: " + "; ".join(f"{r.drill} {r.feature}→{r.outcome} ρ={r.rho:.2f} (p={r.p:.2f})" for r in best.itertuples()) + ". With one attempt per drill and ~60 players, few of these clear p<0.05; this is the baseline the translation analysis has to improve on."),
            "method": "Spearman ρ on best attempt per drill; separation is route-weighted across seasons.",
            "chart": {"type": "table"}, "table": df.sort_values(["drill", "feature"]).to_dict("records")}


def sanitize(o):
    """Recursively turn NaN/inf and numpy scalars into JSON-safe values (strict JSON has no NaN)."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


if __name__ == "__main__":
    d = load()
    p, ev = d["p"], d["combine_events"]
    log(f"players {len(p)}, events {len(ev)}")
    insights = [insight_draft_pays(p), insight_predicts_nfl(p), insight_tracked_vs_official(p), insight_year_effect(ev), insight_reliability(ev), insight_pass_rush(p, ev), insight_wr_breaks(p, ev)]
    pv, ins_val = value_model(p)
    insights += [ins_val, insight_value_lists(pv, p)]
    pv.to_parquet(PROC / "player_value.parquet", index=False)
    (PROC / "insights.json").write_text(json.dumps(sanitize(insights), indent=1))
    for i in insights:
        log(f"{i['id']:22s} {i['title']}")
        print("   ", i["summary"][:400])
    log(f"player_value rows {len(pv)}")
