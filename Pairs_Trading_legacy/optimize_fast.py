"""
Fast Optuna optimizer with crisis protection.
Uses precomputed pair selections + precomputed basket DD indicator.
Goal: maximize CAGR & returns, keep MaxDD above -15%.
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
    INITIAL_CAPITAL, MAX_PAIRS, BUFFER_DAYS,
)

# ============================================================
# LOAD CACHE + PRECOMPUTE CRISIS INDICATOR (once)
# ============================================================
print("Loading cache...")
with open("pair_cache.pkl", "rb") as f:
    CACHE = pickle.load(f)

PRICES = CACHE['prices']
REBALANCE = CACHE['rebalance_schedule']
PAIR_SELECTIONS = CACHE['pair_selections']

# Precompute raw 10-day basket drawdown (fixed, doesn't change per trial)
_log_ret = np.log(PRICES).diff()
_basket_ret = _log_ret.mean(axis=1)
_basket_cum = (1 + _basket_ret.fillna(0)).cumprod()
BASKET_DD_10D = _basket_cum / _basket_cum.rolling(10, min_periods=1).max() - 1

# Pre-slice price windows per (lookback, window_idx) for speed
PRICE_SLICES = {}
for lb, selections in PAIR_SELECTIONS.items():
    for widx, wd in selections.items():
        start, end = wd['start'], wd['end']
        ps = PRICES.loc[start - pd.Timedelta(days=BUFFER_DAYS):end]
        # Pre-extract log prices and index info
        log_prices = {col: np.log(ps[col].values) for col in ps.columns}
        trade_start_idx = ps.index.searchsorted(start)
        PRICE_SLICES[(lb, widx)] = {
            'ps_index': ps.index,
            'log_prices': log_prices,
            'n': len(ps),
            'start': start,
            'end': end,
            'trade_start_idx': trade_start_idx,
            'trade_mask': (ps.index >= start),
        }

print(f"  {PRICES.shape[0]} bars | {len(REBALANCE)} rebalance dates | cache ready")


# ============================================================
# FAST BACKTEST (fully vectorized where possible)
# ============================================================
def fast_backtest(lookback_years, roll_z_window, vol_window,
                  target_daily_vol, max_leverage, txn_cost,
                  stop_z_allow, stop_z_block, min_hold, cooldown,
                  crisis_dd_scale, crisis_dd_halt,
                  entry_z_scale=1.0, exit_z_scale=1.0, n_pairs=10):

    selections = PAIR_SELECTIONS[lookback_years]

    # Compute crisis scale from precomputed DD + trial thresholds
    crisis_raw = BASKET_DD_10D.values
    cs_full = (crisis_raw - crisis_dd_halt) / (crisis_dd_scale - crisis_dd_halt)
    cs_full = np.clip(cs_full, 0.0, 1.0)

    pnl_accum = np.zeros(len(PRICES))
    pair_pnl_dict = {}
    trade_count_total = {}
    cap_per_pair = INITIAL_CAPITAL / n_pairs

    for widx, wd in selections.items():
        pairs = wd['pairs'][:n_pairs]  # use top n_pairs from precomputed
        if not pairs:
            continue

        cache_key = (lookback_years, widx)
        pslice = PRICE_SLICES[cache_key]
        ps_index = pslice['ps_index']
        log_p = pslice['log_prices']
        n = pslice['n']
        start = pslice['start']
        tsi = pslice['trade_start_idx']
        trade_mask = pslice['trade_mask']

        # Get crisis scale for this window
        idx_positions = PRICES.index.searchsorted(ps_index)
        cs_vals = cs_full[idx_positions]

        for a, b, beta, hl in pairs:
            spread = log_p[a] - beta * log_p[b]
            ret = np.empty(n)
            ret[0] = np.nan
            ret[1:] = np.diff(spread)

            # Volatility
            vol = pd.Series(ret).rolling(vol_window).std().values
            vol = np.clip(np.nan_to_num(vol, nan=1e-4), 1e-4, None)

            # Regime filter
            ma = pd.Series(spread).rolling(60).mean().values
            ma_diff = np.empty(n)
            ma_diff[0] = np.nan
            ma_diff[1:] = np.abs(np.diff(ma))
            allow = np.where(np.isnan(ma_diff), False, ma_diff < 2 * vol)

            # Z-score
            s_spread = pd.Series(spread)
            z = (spread - s_spread.rolling(roll_z_window).mean().values) / \
                s_spread.rolling(roll_z_window).std().values

            # Positions (z-score thresholds scaled by tunable multipliers)
            entry_base, exit_base = z_params(hl)
            entry = entry_base * entry_z_scale
            exit_ = exit_base * exit_z_scale
            pos = compute_positions(
                z.astype(np.float64), allow, entry, exit_,
                stop_z_allow, stop_z_block,
                min_hold=min_hold, cooldown=cooldown
            )

            # Trade count (trading window only)
            pair_key = f"{a}-{b}"
            active = pos[tsi:]
            prev = np.empty_like(active)
            prev[0] = 0.0
            prev[1:] = active[:-1]
            entries = int(((active != 0) & (prev == 0)).sum())
            trade_count_total[pair_key] = trade_count_total.get(pair_key, 0) + entries

            # Position sizing — CAUSAL: lag vol and crisis scale by 1 bar
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

            # PnL
            units_diff = np.empty(n)
            units_diff[0] = units[0]
            units_diff[1:] = np.diff(units)
            ppnl = units * ret - np.abs(units_diff) * txn_cost
            ppnl = np.nan_to_num(ppnl, nan=0.0)
            ppnl[:tsi] = 0.0  # zero out buffer

            # Accumulate to global PnL
            global_start = PRICES.index.searchsorted(start)
            global_end = global_start + len(ppnl[tsi:])
            pnl_accum[global_start:global_end] += ppnl[tsi:]

            pair_pnl_dict[pair_key] = pair_pnl_dict.get(pair_key, 0) + ppnl[tsi:].sum()

    pnl = pd.Series(pnl_accum, index=PRICES.index)
    equity = INITIAL_CAPITAL + pnl.cumsum()
    return equity, pnl, pair_pnl_dict, trade_count_total


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def objective(trial):
    params = {
        'lookback_years':   trial.suggest_int("lookback_years", 2, 6),
        'roll_z_window':    trial.suggest_int("roll_z_window", 12, 50),
        'vol_window':       trial.suggest_int("vol_window", 12, 50),
        'target_daily_vol': trial.suggest_float("target_daily_vol", 0.015, 0.065, log=True),
        'max_leverage':     trial.suggest_float("max_leverage", 1.5, 5.0),
        'txn_cost':         trial.suggest_float("txn_cost", 0.0003, 0.002, log=True),
        'stop_z_allow':     trial.suggest_float("stop_z_allow", 2.5, 5.5),
        'stop_z_block':     trial.suggest_float("stop_z_block", 1.2, 3.5),
        'min_hold':         trial.suggest_int("min_hold", 3, 10),
        'cooldown':         trial.suggest_int("cooldown", 2, 8),
        'crisis_dd_scale':  trial.suggest_float("crisis_dd_scale", -0.16, -0.10),
        'crisis_dd_halt':   trial.suggest_float("crisis_dd_halt", -0.22, -0.14),
        'entry_z_scale':    trial.suggest_float("entry_z_scale", 0.6, 1.4),
        'exit_z_scale':     trial.suggest_float("exit_z_scale", 0.5, 1.5),
        'n_pairs':          trial.suggest_int("n_pairs", 4, 10),
    }

    if params['crisis_dd_scale'] < params['crisis_dd_halt']:
        return -1e6

    equity, pnl, _, _ = fast_backtest(**params)
    m = compute_metrics(pnl, equity)

    # Hard constraint: MaxDD below 15%
    if m["Max DD %"] < -15:
        return -1e6
    if m["CAGR %"] < 1:
        return -1e6

    # Objective: maximize CAGR + returns, penalize drawdown
    score = (
        10 * (m["CAGR %"] / 100)
        + 4 * (m["Total Return %"] / 100)
        + 2 * m["Sharpe"]
        - 3 * abs(m["Max DD %"] / 100)
    )
    return score


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    N_TRIALS = 3000

    print(f"\nOptuna: {N_TRIALS} trials | MaxDD constraint: > -15% | Goal: max CAGR + returns")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Warm-start with previous best params
    study.enqueue_trial({
        "lookback_years": 6, "roll_z_window": 15, "vol_window": 31,
        "target_daily_vol": 0.0432, "max_leverage": 2.43, "txn_cost": 0.000337,
        "stop_z_allow": 3.559, "stop_z_block": 1.702, "min_hold": 4, "cooldown": 3,
        "crisis_dd_scale": -0.137, "crisis_dd_halt": -0.150,
        "entry_z_scale": 1.0, "exit_z_scale": 1.0, "n_pairs": 10,
    })

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    elapsed = time.time() - t0

    print(f"\nDone: {elapsed:.1f}s total ({elapsed/N_TRIALS:.2f}s/trial)")
    print(f"Best score: {study.best_value:.4f}")

    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:<15}: {v}")

    # ---- Final backtest ----
    print("\n" + "=" * 60)
    print("FINAL BACKTEST")
    print("=" * 60)

    bp = study.best_params
    equity, pnl, pair_pnl, trade_counts = fast_backtest(
        bp["lookback_years"], bp["roll_z_window"], bp["vol_window"],
        bp["target_daily_vol"], bp["max_leverage"], bp["txn_cost"],
        bp["stop_z_allow"], bp["stop_z_block"], bp["min_hold"], bp["cooldown"],
        bp["crisis_dd_scale"], bp["crisis_dd_halt"],
        bp["entry_z_scale"], bp["exit_z_scale"], bp["n_pairs"],
    )

    save_outputs(equity, pnl, pair_pnl, trade_counts)

    metrics = compute_metrics(pnl, equity)
    print("\nMETRICS")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.2f}")

    print(f"\nTRADES: {sum(trade_counts.values())} total")
    print("\nTOP 5 PAIRS BY PNL")
    print(pairwise_pnl_table(pair_pnl).head(5))

    # Save params
    import os
    os.makedirs("backtest_results", exist_ok=True)
    pd.DataFrame([bp]).to_csv("backtest_results/best_params.csv", index=False)
    print("\nBest params -> backtest_results/best_params.csv")
