"""
Precompute pair selections V2 — PRODUCTION (speed-optimized).
Saves to pair_cache_v2.pkl.
  - Exact same logic as strat.select_pairs (correlation pre-filter, same scoring)
  - Parallel window processing (all CPU cores)
  - Stores MAX_PAIRS=10 per window for optimizer flexibility
  - Uses numpy-only operations where possible
"""
import pickle
import time
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import strat


def select_pairs_fast(log_prices_np, col_names, lookback_years, end_date, prices_index,
                      max_pairs=10):
    """
    Fast pair selection — IDENTICAL logic to strat.select_pairs.
    Uses numpy arrays to avoid pickling DataFrames.
    """
    from statsmodels.tsa.stattools import coint
    from collections import Counter

    start = end_date - pd.DateOffset(years=lookback_years)
    start_idx = prices_index.searchsorted(start)
    end_idx = prices_index.searchsorted(end_date, side='right')
    logp = log_prices_np[start_idx:end_idx]

    if logp.shape[0] < 252:
        return []

    n_stocks = logp.shape[1]

    # Pre-compute pairwise correlations for filtering (matches strat.py L287-292)
    returns = np.diff(logp, axis=0)
    if returns.shape[0] < 100:
        return []
    corr_matrix = np.corrcoef(returns.T)

    candidates = []
    for i in range(n_stocks):
        for j in range(i + 1, n_stocks):
            # PRE-FILTER: skip low-correlation pairs (matches strat.py L297-299)
            if abs(corr_matrix[i, j]) < 0.3:
                continue

            x, y = logp[:, i], logp[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 252:
                continue
            x, y = x[mask], y[mask]

            try:
                t_stat, pval, _ = coint(x, y)
            except Exception:
                continue
            if pval > 0.05:
                continue

            beta = strat.hedge_ratio_nb(x, y)
            spread = x - beta * y
            hl = strat.half_life_nb(spread)
            if hl < 3 or hl > 60:
                continue

            corr = strat.rolling_corr_nb(x, y, 126)
            beta_std = strat.rolling_beta_std(x, y)
            if not np.isfinite(beta_std):
                continue

            corr_mean = np.nanmean(corr)
            corr_std = np.nanstd(corr)

            # Scoring — identical to strat.py L328-334
            score = (
                2 * abs(t_stat)
                + 1.5 * abs(corr_mean)
                - corr_std
                - 2 * beta_std
                - 0.1 * abs(hl - 15)
            )
            candidates.append((col_names[i], col_names[j], beta, hl, score))

    ranked = sorted(candidates, key=lambda x: x[-1], reverse=True)

    # Selection with usage constraint — identical to strat.py L343-350
    selected, used = [], Counter()
    for a, b, beta, hl, _ in ranked:
        if used[a] < 2 and used[b] < 2:
            selected.append((a, b, beta, hl))
            used[a] += 1
            used[b] += 1
        if len(selected) == max_pairs:
            break

    return selected


def process_window(args):
    """Worker function for a single rebalance window."""
    widx, start_ts, end_ts, lookback_years, log_prices_np, col_names, prices_index, max_pairs = args
    start = pd.Timestamp(start_ts)
    end = pd.Timestamp(end_ts)
    pairs = select_pairs_fast(log_prices_np, col_names, lookback_years, start,
                              prices_index, max_pairs)
    return widx, {
        'start': start,
        'end': end,
        'pairs': pairs if pairs else [],
    }


def main():
    print("=" * 60)
    print("PRECOMPUTING PAIR SELECTIONS V2 (PRODUCTION)")
    print("=" * 60)

    strat.init_data()
    prices = strat.GLOBAL_PRICES
    rebalance = strat.GLOBAL_REBALANCE

    print(f"  Prices shape: {prices.shape}")
    print(f"  Rebalance dates: {len(rebalance)}")
    print(f"  MAX_PAIRS stored: 10 (optimizer picks top N)")
    print(f"  CPU cores: {os.cpu_count()}")

    log_prices_np = np.log(prices.values)
    col_names = list(prices.columns)
    prices_index = prices.index

    cache = {
        'prices': prices,
        'rebalance_schedule': rebalance,
        'pair_selections': {},
    }

    lookback_range = range(2, 7)
    n_workers = min(os.cpu_count() or 4, 8)

    for lb in lookback_range:
        print(f"\n--- Lookback = {lb} years (parallel, {n_workers} workers) ---")
        t_lb = time.time()

        window_args = []
        for i in range(1, len(rebalance)):
            window_args.append((
                i,
                rebalance[i - 1].isoformat(),
                rebalance[i].isoformat(),
                lb,
                log_prices_np,
                col_names,
                prices_index,
                10,  # always store top 10 for optimizer flexibility
            ))

        selections = {}
        completed = 0
        total = len(window_args)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_window, args): args[0]
                       for args in window_args}
            for future in as_completed(futures):
                widx, result = future.result()
                selections[widx] = result
                completed += 1
                if completed % 40 == 0 or completed == total:
                    print(f"  {completed}/{total} windows done ({time.time()-t_lb:.1f}s)")

        cache['pair_selections'][lb] = selections
        print(f"  Lookback {lb}y done in {time.time()-t_lb:.1f}s")

    cache_path = "pair_cache_v2.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nCache saved to: {cache_path}")
    for lb in lookback_range:
        total_pairs = sum(len(s['pairs']) for s in cache['pair_selections'][lb].values())
        print(f"  Lookback {lb}y: {total_pairs} total pair-slots across {len(cache['pair_selections'][lb])} windows")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal precompute time: {time.time() - t0:.1f}s")
