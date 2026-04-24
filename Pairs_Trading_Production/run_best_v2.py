"""
Run the best params from best_params_v2.csv using the V2 production cost model.
Quick replay without re-running Optuna.
"""
import pandas as pd
from optimize_fast_v2 import fast_backtest, PRICES
from strat import compute_metrics, save_outputs, pairwise_pnl_table, INITIAL_CAPITAL

print("=" * 60)
print("RUNNING FAST BACKTEST ON BEST PARAMS (V2 PRODUCTION)")
print("=" * 60)

bp = pd.read_csv("backtest_results/best_params_v2.csv").iloc[0].to_dict()

# Convert integer params
int_keys = ['lookback_years', 'roll_z_window', 'vol_window', 'min_hold', 'cooldown', 'n_pairs']
for k in int_keys:
    bp[k] = int(bp[k])

print("Loaded parameters:")
for k, v in bp.items():
    print(f"  {k:<20}: {v}")

print("\nRunning backtest from cache...")
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
print("\nOutputs saved to backtest_results/ folder.")
