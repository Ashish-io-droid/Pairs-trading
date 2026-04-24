import pandas as pd
from optimize_fast import fast_backtest, compute_metrics, save_outputs, pairwise_pnl_table

def run_best():
    print("=" * 60)
    print("RUNNING FAST BACKTEST ON BEST PARAMS")
    print("=" * 60)

    try:
        df = pd.read_csv("backtest_results/best_params.csv")
        bp = df.iloc[0].to_dict()
    except Exception as e:
        print("Could not load best_params.csv. Have you run the optimizer yet?")
        return

    print("Loaded parameters:")
    for k, v in bp.items():
        print(f"  {k:<15}: {v}")

    # Ensure types are correct for integer parameters
    bp["lookback_years"] = int(bp["lookback_years"])
    bp["roll_z_window"] = int(bp["roll_z_window"])
    bp["vol_window"] = int(bp["vol_window"])
    bp["min_hold"] = int(bp["min_hold"])
    bp["cooldown"] = int(bp["cooldown"])
    bp["n_pairs"] = int(bp.get("n_pairs", 5))

    print("\nRunning backtest from cache...")
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
    print("\nOutputs saved to backtest_results/ folder.")

if __name__ == "__main__":
    run_best()
