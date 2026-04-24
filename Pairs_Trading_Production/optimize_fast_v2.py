"""
Fast Optuna optimizer V2 — PRODUCTION EXECUTION SIMULATOR.
Uses pair_cache_v2.pkl with the full production cost model.

OBJECTIVE: Maximize CAGR with MaxDD > -15% hard constraint.

Speed optimizations:
  - Pre-sliced numpy arrays (no dict lookups per pair)
  - Pre-computed cost rates (avoid per-bar division)
  - Minimal pandas usage inside hot loop
  - Pre-allocated output arrays

Logic: IDENTICAL to strat.py trade_pairs (lines 383-463).
"""
import pickle
import numpy as np
import pandas as pd
import optuna
import time
import warnings
warnings.filterwarnings("ignore")

from strat import (
    compute_positions, z_params, compute_metrics,
    save_outputs, pairwise_pnl_table,
    INITIAL_CAPITAL, BUFFER_DAYS,
)

# ============================================================
# LOAD CACHE + PRECOMPUTE (once at import)
# ============================================================
CACHE_PATH = "pair_cache_v2.pkl"
print(f"Loading cache from {CACHE_PATH}...")
with open(CACHE_PATH, "rb") as f:
    CACHE = pickle.load(f)

PRICES = CACHE['prices']
REBALANCE = CACHE['rebalance_schedule']
PAIR_SELECTIONS = CACHE['pair_selections']

# Precompute raw 10-day basket drawdown (fixed, doesn't change per trial)
_log_ret = np.log(PRICES).diff()
_basket_ret = _log_ret.mean(axis=1)
_basket_cum = (1 + _basket_ret.fillna(0)).cumprod()
_rolling_peak = _basket_cum.rolling(10, min_periods=1).max()
BASKET_DD_10D = (_basket_cum / _rolling_peak - 1).values  # numpy array for speed

# Pre-slice price windows — use NUMPY 2D arrays instead of dicts for speed
COL_INDEX = {col: idx for idx, col in enumerate(PRICES.columns)}
PRICE_SLICES = {}

for lb, selections in PAIR_SELECTIONS.items():
    for widx, wd in selections.items():
        start, end = wd['start'], wd['end']
        ps = PRICES.loc[start - pd.Timedelta(days=BUFFER_DAYS):end]

        # Store as contiguous numpy 2D array (much faster than dict of 1D arrays)
        log_prices_2d = np.log(ps.values).astype(np.float64)
        tsi = ps.index.searchsorted(start)
        idx_positions = PRICES.index.searchsorted(ps.index)

        PRICE_SLICES[(lb, widx)] = {
            'log_prices_2d': log_prices_2d,  # shape (n_bars, n_stocks)
            'n': len(ps),
            'start': start,
            'tsi': tsi,
            'crisis_idx': idx_positions,      # map to global BASKET_DD_10D
        }

print(f"  {PRICES.shape[0]} bars | {len(REBALANCE)} rebalance dates | cache ready")


# ============================================================
# FAST BACKTEST — PRODUCTION COST MODEL
# ============================================================
# Logic mapping to strat.py:
#   spread = x - beta*y                           → strat.py L387
#   ret = np.diff(spread, prepend=np.nan)          → strat.py L388
#   vol = rolling(VOL_WINDOW).std, clip(1e-4)      → strat.py L390-391
#   ma = rolling(60).mean, allow = ma_diff < 2*vol → strat.py L394-396
#   z = (spread - rolling_mean) / rolling_std      → strat.py L402-403
#   pos = compute_positions(z, allow, ...)         → strat.py L410-412
#   lagged_pos = [0, pos[:-1]]                     → strat.py L421
#   vol_lagged = [vol[0], vol[:-1]]                → strat.py L422
#   cs_lagged = [1.0, cs_vals[:-1]]                → strat.py L424
#   units = cap * risk * lagged_pos * cs_lagged    → strat.py L425
#   costs = brokerage + slippage + impact + borrow → strat.py L427-444
#   ppnl = units * ret - total_cost                → strat.py L447
# ============================================================

def fast_backtest(lookback_years, roll_z_window, vol_window,
                  target_daily_vol, max_leverage,
                  brokerage_bps, slippage_bps, market_impact_bps, short_borrow_rate,
                  stop_z_allow, stop_z_block, min_hold, cooldown,
                  crisis_dd_scale, crisis_dd_halt,
                  entry_z_scale=1.0, exit_z_scale=1.0, n_pairs=4):

    selections = PAIR_SELECTIONS[lookback_years]

    # Compute crisis scale from precomputed DD + trial thresholds
    denom = crisis_dd_scale - crisis_dd_halt
    if abs(denom) < 1e-10:
        cs_full = np.ones(len(PRICES))
    else:
        cs_full = np.clip((BASKET_DD_10D - crisis_dd_halt) / denom, 0.0, 1.0)

    pnl_accum = np.zeros(len(PRICES))
    pair_pnl_dict = {}
    trade_count_total = {}
    cost_total = np.zeros(4)  # [brokerage, slippage, impact, borrow]
    cap_per_pair = INITIAL_CAPITAL / n_pairs

    # Pre-compute cost rates (avoid per-bar division)
    brokerage_rate = brokerage_bps / 10_000
    slippage_rate = slippage_bps * 2 / 10_000      # per leg × 2 legs
    impact_rate = market_impact_bps * 2 / 10_000    # per leg × 2 legs
    borrow_daily = short_borrow_rate * 0.5 / 252    # 50% notional is short leg

    # Combined trade-cost rate (brokerage + slippage + impact applied to |Δunits|)
    trade_cost_rate = brokerage_rate + slippage_rate + impact_rate

    global_start_cache = {}  # cache searchsorted results

    for widx, wd in selections.items():
        pairs = wd['pairs'][:n_pairs]
        if not pairs:
            continue

        cache_key = (lookback_years, widx)
        pslice = PRICE_SLICES[cache_key]
        log_p_2d = pslice['log_prices_2d']
        n = pslice['n']
        start = pslice['start']
        tsi = pslice['tsi']
        crisis_idx = pslice['crisis_idx']

        # Get crisis scale for this window
        cs_vals = cs_full[crisis_idx]

        # Cache the global start index
        if start not in global_start_cache:
            global_start_cache[start] = PRICES.index.searchsorted(start)
        global_start = global_start_cache[start]

        for a, b, beta, hl in pairs:
            # Column indices for numpy 2D array lookup
            ai, bi = COL_INDEX[a], COL_INDEX[b]

            # Spread & returns — matches strat.py L385-388
            spread = log_p_2d[:, ai] - beta * log_p_2d[:, bi]
            ret = np.empty(n)
            ret[0] = np.nan
            ret[1:] = np.diff(spread)

            # Volatility — matches strat.py L390-391
            vol = pd.Series(ret).rolling(vol_window).std().values
            vol = np.clip(np.nan_to_num(vol, nan=1e-4), 1e-4, None)

            # Regime filter — matches strat.py L394-396
            ma = pd.Series(spread).rolling(60).mean().values
            ma_diff = np.empty(n)
            ma_diff[0] = np.nan
            ma_diff[1:] = np.abs(np.diff(ma))
            allow = np.where(np.isnan(ma_diff), False, ma_diff < 2 * vol)

            # Z-score — matches strat.py L402-403
            s_spread = pd.Series(spread)
            z_mean = s_spread.rolling(roll_z_window).mean().values
            z_std = s_spread.rolling(roll_z_window).std().values
            z = (spread - z_mean) / z_std

            # Positions — matches strat.py L405-412
            entry_base, exit_base = z_params(hl)
            entry = entry_base * entry_z_scale
            exit_ = exit_base * exit_z_scale
            pos = compute_positions(
                z.astype(np.float64), allow, entry, exit_,
                stop_z_allow, stop_z_block,
                min_hold=min_hold, cooldown=cooldown
            )

            # Trade count (trading window only) — matches strat.py L414-418
            pair_key = f"{a}-{b}"
            active = pos[tsi:]
            prev = np.empty_like(active)
            prev[0] = 0.0
            prev[1:] = active[:-1]
            entries = int(((active != 0) & (prev == 0)).sum())
            trade_count_total[pair_key] = trade_count_total.get(pair_key, 0) + entries

            # ---- EXECUTION (all lagged) — matches strat.py L420-425 ----
            lagged_pos = np.empty(n)
            lagged_pos[0] = 0.0
            lagged_pos[1:] = pos[:-1]

            vol_lagged = np.empty(n)
            vol_lagged[0] = vol[0]
            vol_lagged[1:] = vol[:-1]

            risk = np.clip(target_daily_vol / vol_lagged, 0, max_leverage)

            cs_lagged = np.empty(n)
            cs_lagged[0] = 1.0
            cs_lagged[1:] = cs_vals[:-1]

            units = cap_per_pair * risk * lagged_pos * cs_lagged

            # ---- PRODUCTION COST MODEL — matches strat.py L427-444 ----
            units_diff = np.empty(n)
            units_diff[0] = units[0]
            units_diff[1:] = np.diff(units)
            abs_traded = np.abs(units_diff)
            abs_units = np.abs(units)

            # Combined trade costs (brokerage + slippage + impact on |Δunits|)
            trade_costs = abs_traded * trade_cost_rate
            # Borrow cost (daily accrual on |units|)
            borrow_costs = abs_units * borrow_daily
            # Total
            total_cost = trade_costs + borrow_costs

            # PnL — matches strat.py L447
            ppnl = units * ret - total_cost
            ppnl = np.nan_to_num(ppnl, nan=0.0)
            ppnl[:tsi] = 0.0  # zero out buffer — matches strat.py L450-451

            # Accumulate cost breakdown (trading window only)
            tw_traded = abs_traded[tsi:]
            cost_total[0] += (tw_traded * brokerage_rate).sum()
            cost_total[1] += (tw_traded * slippage_rate).sum()
            cost_total[2] += (tw_traded * impact_rate).sum()
            cost_total[3] += (abs_units[tsi:] * borrow_daily).sum()

            # Accumulate to global PnL
            trade_pnl = ppnl[tsi:]
            global_end = global_start + len(trade_pnl)
            pnl_accum[global_start:global_end] += trade_pnl

            pair_pnl_dict[pair_key] = pair_pnl_dict.get(pair_key, 0) + trade_pnl.sum()

    pnl = pd.Series(pnl_accum, index=PRICES.index)
    equity = INITIAL_CAPITAL + pnl.cumsum()

    costs_dict = {
        "brokerage": cost_total[0],
        "slippage": cost_total[1],
        "impact": cost_total[2],
        "borrow": cost_total[3],
    }
    return equity, pnl, pair_pnl_dict, trade_count_total, costs_dict


# ============================================================
# OPTUNA OBJECTIVE — MAXIMIZE CAGR, MaxDD > -15%
# ============================================================
def objective(trial):
    params = {
        'lookback_years':     trial.suggest_int("lookback_years", 2, 6),
        'roll_z_window':      trial.suggest_int("roll_z_window", 10, 50),
        'vol_window':         trial.suggest_int("vol_window", 10, 50),
        'target_daily_vol':   trial.suggest_float("target_daily_vol", 0.015, 0.080, log=True),
        'max_leverage':       trial.suggest_float("max_leverage", 1.5, 5.0),
        # Production cost params
        'brokerage_bps':      trial.suggest_float("brokerage_bps", 3.0, 8.0),
        'slippage_bps':       trial.suggest_float("slippage_bps", 2.0, 8.0),
        'market_impact_bps':  trial.suggest_float("market_impact_bps", 1.0, 5.0),
        'short_borrow_rate':  trial.suggest_float("short_borrow_rate", 0.005, 0.04),
        # Signal params
        'stop_z_allow':       trial.suggest_float("stop_z_allow", 2.0, 6.0),
        'stop_z_block':       trial.suggest_float("stop_z_block", 1.0, 3.5),
        'min_hold':           trial.suggest_int("min_hold", 3, 12),
        'cooldown':           trial.suggest_int("cooldown", 1, 6),
        'crisis_dd_scale':    trial.suggest_float("crisis_dd_scale", -0.18, -0.08),
        'crisis_dd_halt':     trial.suggest_float("crisis_dd_halt", -0.25, -0.12),
        'entry_z_scale':      trial.suggest_float("entry_z_scale", 0.5, 1.5),
        'exit_z_scale':       trial.suggest_float("exit_z_scale", 0.4, 1.8),
        'n_pairs':            trial.suggest_int("n_pairs", 4, 8),
    }

    # Sanity: crisis_dd_scale must be > crisis_dd_halt (less negative)
    if params['crisis_dd_scale'] < params['crisis_dd_halt']:
        return -1e6

    equity, pnl, _, _, _ = fast_backtest(**params)
    m = compute_metrics(pnl, equity)

    # ---- HARD CONSTRAINTS ----
    if m["Max DD %"] < -15:
        return -1e6
    if m["CAGR %"] < 1:
        return -1e6

    # ---- OBJECTIVE: Maximize CAGR + returns, penalize drawdown ----
    score = (
        10 * (m["CAGR %"] / 100)             # primary: maximize CAGR
        + 4 * (m["Total Return %"] / 100)     # reward total return
        + 2 * m["Sharpe"]                     # reward risk-adjusted return
        - 3 * abs(m["Max DD %"] / 100)        # strong drawdown penalty
    )
    return score


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    N_TRIALS = 3000

    print("=" * 60)
    print(f"PRODUCTION OPTIMIZER V2 — {N_TRIALS} trials")
    print("=" * 60)
    print("Objective : Maximize CAGR")
    print("Constraint: MaxDD > -15%")
    print("Cost model: brokerage + slippage(2-leg) + impact(2-leg) + borrow")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Warm-start with current best params from strat.py
    study.enqueue_trial({
        "lookback_years": 5, "roll_z_window": 17, "vol_window": 19,
        "target_daily_vol": 0.062, "max_leverage": 3.16,
        "brokerage_bps": 4.77, "slippage_bps": 5.0,
        "market_impact_bps": 2.0, "short_borrow_rate": 0.02,
        "stop_z_allow": 2.69, "stop_z_block": 2.14,
        "min_hold": 7, "cooldown": 2,
        "crisis_dd_scale": -0.102, "crisis_dd_halt": -0.159,
        "entry_z_scale": 0.72, "exit_z_scale": 1.32,
        "n_pairs": 4,
    })

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed = time.time() - t0

    print(f"\nDone: {elapsed:.1f}s total ({elapsed/N_TRIALS:.2f}s/trial)")
    print(f"Best score: {study.best_value:.4f}")

    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:<20}: {v}")

    # ---- Final backtest ----
    print("\n" + "=" * 60)
    print("FINAL BACKTEST — PRODUCTION COST MODEL")
    print("=" * 60)

    bp = study.best_params
    equity, pnl, pair_pnl, trade_counts, costs = fast_backtest(**bp)

    save_outputs(equity, pnl, pair_pnl, trade_counts)

    metrics = compute_metrics(pnl, equity)
    print("\nMETRICS")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.2f}")

    print(f"\nEXECUTION COST BREAKDOWN")
    total_cost = sum(costs.values())
    if total_cost > 0:
        print(f"  Brokerage      : ₹{costs['brokerage']:>12,.2f}  ({100*costs['brokerage']/total_cost:.1f}%)")
        print(f"  Slippage (2-leg): ₹{costs['slippage']:>12,.2f}  ({100*costs['slippage']/total_cost:.1f}%)")
        print(f"  Market Impact   : ₹{costs['impact']:>12,.2f}  ({100*costs['impact']/total_cost:.1f}%)")
        print(f"  Short Borrow    : ₹{costs['borrow']:>12,.2f}  ({100*costs['borrow']/total_cost:.1f}%)")
        print(f"  ─────────────────────────────────────")
        print(f"  TOTAL FRICTION  : ₹{total_cost:>12,.2f}")
        print(f"  As % of Capital : {100*total_cost/INITIAL_CAPITAL:.2f}%")

    print(f"\nTRADES: {sum(trade_counts.values())} total")
    print("\nTOP 5 PAIRS BY PNL")
    print(pairwise_pnl_table(pair_pnl).head(5))

    # Save params
    import os
    os.makedirs("backtest_results", exist_ok=True)
    pd.DataFrame([bp]).to_csv("backtest_results/best_params_v2.csv", index=False)
    print("\nBest params -> backtest_results/best_params_v2.csv")
