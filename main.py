#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun bounty roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (chart snapshots) + /input/task.json, writes
/output/forecast.json: one static per-place probability forecast + top10,
scored against every hourly read of the 72h window after entries close.

Method (stdlib only, fully deterministic, no network, no wall clock).
Two independent rank forecasts are blended 50/50 per place:
  A. Gaussian drift: per game, rank velocity by least squares over the last
     ~4 days (+ a playing-growth boost), extrapolated damped to the window
     midpoint; uncertainty grows with horizon, velocity and rank volatility.
     Probability at a place is a Gaussian kernel around the forecast rank.
  B. Empirical transitions: a binned rank-transition matrix learned from all
     snapshot pairs about one horizon apart (smoothed toward persistence),
     giving P(future rank | current rank) directly from the chart's own
     churn. Each game's place distribution is the row for its
     (drift-adjusted) last rank.
Each place keeps 0.30 of its mass unnamed, so surprise entrants still score
above the floor; top10 = the 10 games most likely to be in the top 10.

Backtested on the dev history (log score, higher is better):
  train<=Nov 4, test Nov 5-7: model -39.6 vs persistence -55.8
  train<=Nov 2, test Nov 3-7: model -44.4 vs persistence -55.7
"""
import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunables (chosen by backtesting on the dev history) ----
VEL_DAYS = 4.0        # lookback window (days) for rank velocity
VEL_DAMP = 0.3        # shrink extrapolated drift toward no-change
VEL_CAP_DAYS = 4.0    # never extrapolate more than this many days of drift
PLAY_BOOST = 12.0     # ranks/day equivalent per unit log-playing slope
SIGMA_BASE = 2.5      # rank uncertainty at horizon 0
SIGMA_PER_DAY = 2.5   # extra rank uncertainty per day of horizon
SIGMA_VOL = 0.8       # weight of historical rank volatility in sigma
UNNAMED = 0.30        # probability mass left unnamed at each place
MIN_P = 0.001         # drop named rows below this (they join the unnamed pool)
CAND_UC = 80          # candidate if last up-and-coming rank <= this
CAND_TT = 60          # candidate if top-trending rank <= this
CAND_TP = 60          # candidate if top-playing-now rank <= this
CAND_RHAT = 25        # candidate if forecast rank <= this
BLEND_EMP = 0.5       # weight of the empirical forecast in the blend
TRANS_TOL_DAYS = 0.75 # snapshot-pair gap tolerance around the horizon
TRANS_PRIOR = 15.0    # smoothing prior weight (persistence) per matrix cell
TRANS_PRIOR_SIGMA = 2.0  # width (bins) of the persistence prior
# rank bins for the transition matrix: 1..10 single, then bands to 101
EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 31, 51, 102]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("ok", True):
                continue
            try:
                t = parse_time(r["at"])
            except (KeyError, ValueError):
                continue
            rows.append((t, r))
    rows.sort(key=lambda x: x[0])
    return rows


def chart_map(snapshot, key):
    """universe_id -> rank for one chart list of a snapshot ({} if absent)."""
    out = {}
    for row in snapshot.get(key) or []:
        try:
            out[int(row["universe_id"])] = int(row["rank"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def lin_slope(xs, ys):
    """Least-squares slope of y on x; 0.0 when degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def bin_of(rank):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= rank < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def build_series(rows, t0):
    """Per-game time series from the up-and-coming chart (+ cross charts)."""
    games = {}
    for t, snap in rows:
        days = (t - t0).total_seconds() / 86400.0
        uc = chart_map(snap, "up_and_coming")
        tt = chart_map(snap, "top_trending")
        tp = chart_map(snap, "top_playing_now")
        playing = {}
        for row in snap.get("up_and_coming") or []:
            try:
                playing[int(row["universe_id"])] = float(row.get("playing") or 0)
            except (KeyError, TypeError, ValueError):
                continue
        for uid in set(uc) | set(tt) | set(tp):
            g = games.setdefault(
                uid, {"t": [], "rank": [], "playing": [], "tt": None, "tp": None}
            )
            g["t"].append(days)
            g["rank"].append(float(uc.get(uid, 101)))
            g["playing"].append(playing.get(uid, 0.0))
        for uid, r in tt.items():
            if uid in games:
                games[uid]["tt"] = r
        for uid, r in tp.items():
            if uid in games:
                games[uid]["tp"] = r
    return games


def game_forecast(g, horizon_days):
    """(forecast rank, sigma) for one game's series at the horizon."""
    last_rank = g["rank"][-1]
    last_playing = g["playing"][-1]
    pts = [(t, r) for t, r in zip(g["t"], g["rank"]) if t >= -VEL_DAYS]
    if len(pts) >= 3:
        v = lin_slope([p[0] for p in pts], [p[1] for p in pts])
        v *= len(pts) / (len(pts) + 2.0)
    elif len(pts) == 2:
        v = (pts[1][1] - pts[0][1]) / max(0.25, pts[1][0] - pts[0][0])
        v *= 0.5
    else:
        v = 0.0
    v = max(-25.0, min(25.0, v))
    steps = []
    for i in range(1, len(g["rank"])):
        dt = g["t"][i] - g["t"][i - 1]
        if dt > 0:
            steps.append(abs(g["rank"][i] - g["rank"][i - 1]) / dt)
    vol = sum(steps) / len(steps) if steps else 5.0
    vol = min(vol, 30.0)
    lp = [(t, math.log(p)) for t, p in zip(g["t"], g["playing"]) if p > 0 and t >= -VEL_DAYS]
    play_boost = 0.0
    if len(lp) >= 3:
        ps = lin_slope([p[0] for p in lp], [p[1] for p in lp])
        play_boost = max(-1.5, min(1.5, ps)) * PLAY_BOOST
    drift_days = min(horizon_days, VEL_CAP_DAYS)
    r_hat = last_rank + (v - play_boost) * drift_days * VEL_DAMP
    if g["tp"] is not None and g["tp"] <= 20 and last_rank > 30:
        r_hat -= 4.0
    r_hat = max(1.0, min(101.0, r_hat))
    first_seen_days = -g["t"][0]
    sigma = SIGMA_BASE + SIGMA_PER_DAY * horizon_days + SIGMA_VOL * min(vol, 15.0)
    if first_seen_days < 2.0:
        sigma += 3.0
    if last_playing <= 0:
        sigma += 2.0
    return r_hat, sigma


def transition_matrix(rows, horizon_days):
    """P(future rank bin | current rank bin) from snapshot pairs ~horizon apart."""
    nb = len(EDGES) - 1
    counts = [[0.0] * nb for _ in range(nb)]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            gap = (rows[j][0] - rows[i][0]).total_seconds() / 86400.0
            if abs(gap - horizon_days) > TRANS_TOL_DAYS:
                continue
            a = chart_map(rows[i][1], "up_and_coming")
            b = chart_map(rows[j][1], "up_and_coming")
            for uid, ra in a.items():
                counts[bin_of(ra)][bin_of(b.get(uid, 101))] += 1.0
    for r in range(nb):
        for c in range(nb):
            d = abs(c - r)
            counts[r][c] += TRANS_PRIOR * math.exp(-0.5 * (d / TRANS_PRIOR_SIGMA) ** 2)
    return [[x / sum(counts[r]) for x in counts[r]] for r in range(nb)]


def forecast(rows, task):
    if not rows:
        return {"probabilities": [], "top10": []}
    t0 = rows[-1][0]
    window = task.get("window", {}) or {}
    try:
        w_from = parse_time(window["from"])
        w_until = parse_time(window["until"])
    except (KeyError, ValueError):
        w_from = w_until = t0
    mid = w_from + (w_until - w_from) / 2
    horizon_days = max(0.0, (mid - t0).total_seconds() / 86400.0)

    games = build_series(rows, t0)
    trans = transition_matrix(rows, horizon_days)

    # per-game forecast rank + sigma; candidate set
    info = {}  # uid -> (r_hat, sigma)
    for uid in sorted(games):
        g = games[uid]
        r_hat, sigma = game_forecast(g, horizon_days)
        last_rank = g["rank"][-1]
        if (
            last_rank <= CAND_UC
            or (g["tt"] is not None and g["tt"] <= CAND_TT)
            or (g["tp"] is not None and g["tp"] <= CAND_TP)
            or r_hat <= CAND_RHAT
        ):
            info[uid] = (r_hat, sigma)

    # ---- forecast A: Gaussian kernel around the forecast rank ----
    gauss = {}  # (uid, place) -> p
    for place in range(1, 11):
        weights = []
        for uid, (r_hat, sigma) in info.items():
            z = (r_hat - place) / sigma
            weights.append((uid, math.exp(-0.5 * z * z)))
        total = sum(w for _, w in weights)
        if total <= 0:
            continue
        kept = []
        named = 0.0
        for uid, w in weights:
            p = (w / total) * (1.0 - UNNAMED)
            if p >= MIN_P:
                kept.append((uid, p))
                named += p
        if kept and named > 0:
            scale = (1.0 - UNNAMED) / named
            for uid, p in kept:
                gauss[(uid, place)] = p * scale

    # ---- forecast B: empirical transition row for the drift-adjusted rank ----
    # Only games on the last up-and-coming chart: the transition matrix is
    # learned on that chart, and games off it dilute the distribution.
    last_uc = chart_map(rows[-1][1], "up_and_coming")
    # The empirical row uses a plain drifted rank (least-squares velocity over
    # the last 4 days, damped); the playing-boost / shrinkage of the Gaussian
    # part stay out of it - backtests show the plain drift scores better here.
    emp = {}  # (uid, place) -> p
    for place in range(1, 11):
        weights = []
        for uid in last_uc:
            g = games[uid]
            pts = [(t, r) for t, r in zip(g["t"], g["rank"]) if t >= -VEL_DAYS]
            if len(pts) >= 3:
                v = lin_slope([p[0] for p in pts], [p[1] for p in pts])
            else:
                v = 0.0
            v = max(-25.0, min(25.0, v))
            r_adj = max(1.0, min(101.0, g["rank"][-1] + v * min(horizon_days, VEL_CAP_DAYS) * VEL_DAMP))
            p = trans[bin_of(r_adj)][place - 1]
            if p >= MIN_P:
                weights.append((uid, p))
        total = sum(w for _, w in weights)
        if total <= 0:
            continue
        kept = []
        named = 0.0
        for uid, w in weights:
            p = (w / total) * (1.0 - UNNAMED)
            if p >= MIN_P:
                kept.append((uid, p))
                named += p
        if kept and named > 0:
            scale = (1.0 - UNNAMED) / named
            for uid, p in kept:
                emp[(uid, place)] = p * scale

    # ---- blend 50/50 per place, renormalised to (1 - UNNAMED) ----
    probabilities = []
    p_top10 = {}
    for place in range(1, 11):
        keys = {k for k in set(gauss) | set(emp) if k[1] == place}
        if not keys:
            continue
        raw = {k: (1.0 - BLEND_EMP) * gauss.get(k, 0.0) + BLEND_EMP * emp.get(k, 0.0) for k in keys}
        total = sum(raw.values())
        if total <= 0:
            continue
        for k in sorted(keys, key=lambda kk: (-raw[kk], kk[0])):
            p = raw[k] / total * (1.0 - UNNAMED)
            if p >= MIN_P:
                probabilities.append({"universe_id": k[0], "place": place, "p": round(p, 6)})
                p_top10[k[0]] = p_top10.get(k[0], 0.0) + p
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # rank for tie-breaking: forecast rank where known, else last chart rank
    def tie_rank(uid):
        if uid in info:
            return info[uid][0]
        g = games.get(uid)
        return g["rank"][-1] if g else 101.0

    ranked = sorted(p_top10.items(), key=lambda kv: (-kv[1], tie_rank(kv[0]), kv[0]))
    top10 = [uid for uid, _ in ranked[:10]]
    if len(top10) < 10:
        rest = sorted(info.items(), key=lambda kv: (kv[1][0], kv[0]))
        for uid, _ in rest:
            if uid not in top10:
                top10.append(uid)
            if len(top10) == 10:
                break
    return {"probabilities": probabilities, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    rows = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    result = forecast(rows, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
