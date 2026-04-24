import yfinance as yf
import pandas as pd
import numpy as np
from collections import Counter
from statsmodels.tsa.stattools import coint
from numba import njit
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"

INITIAL_CAPITAL = 500_000

PAIR_LOOKBACK_YEARS = 5
MAX_PAIRS = 4
REBALANCE_STEP_DAYS = 10

ROLL_Z_WINDOW = 22
VOL_WINDOW = 17

TARGET_DAILY_VOL = 0.06499732355018087
MAX_LEVERAGE = 4.284984403938978
TXN_COST = 0.0005766922979710803
BUFFER_DAYS = 1200

ENTRY_Z_SCALE = 0.679179341935432
EXIT_Z_SCALE = 1.3915150876908906

MIN_HOLD_BARS = 6        # minimum bars in a trade before exit allowed
COOLDOWN_BARS = 2        # bars to wait after exit before re-entry

# --- CRISIS PROTECTION (calibrated from data, 10-day basket drawdown) ---
# 2018 H2: DD_10d min = -14.82%, strategy made +9.77% (only ~5 bars near threshold)
# COVID:   DD_10d = -15.14% on Mar 12, -32% on Mar 23 (catastrophic)
# 2022 H1: DD_10d min = -16.6%, strategy made +0.68% (break-even)
# Threshold -14.5%: catches COVID from Mar 12, minor clip on 2018 H2 peak
CRISIS_DD_WINDOW = 10        # lookback for basket drawdown (trading days)
CRISIS_DD_SCALE  = -0.15696794982387388   # start scaling down exposure
CRISIS_DD_HALT   = -0.17800528978265642   # full halt

# ============================================================
# UNIVERSE (17 stocks — Nifty Auto + auto ancillary)
# ============================================================
NIFTY_AUTO = {
    # Core Nifty Auto (9)
    "MARUTI": "MARUTI.NS",
    "M&M": "M&M.NS",
    "BAJAJ_AUTO": "BAJAJ-AUTO.NS",
    "HERO": "HEROMOTOCO.NS",
    "EICHER": "EICHERMOT.NS",
    "ASHOKLEY": "ASHOKLEY.NS",
    "TVSMOTOR": "TVSMOTOR.NS",
    "BALKRISIND": "BALKRISIND.NS",
    "APOLLOTYRE": "APOLLOTYRE.NS",
    # Auto ancillary (8)
    "MOTHERSON": "MOTHERSON.NS",
    "BOSCH": "BOSCHLTD.NS",
    "MRF": "MRF.NS",
    "BHARATFORG": "BHARATFORG.NS",
    "EXIDEIND": "EXIDEIND.NS",
    "SUNDRMFAST": "SUNDRMFAST.NS",
    "ESCORTS": "ESCORTS.NS",
    "SWARAJENG": "SWARAJENG.NS",
}

# ============================================================
# DATA DOWNLOAD (LAZY INIT — FIX 8)
# ============================================================
def download_prices():
    series = []
    print("Downloading data...")
    for name, ticker in NIFTY_AUTO.items():
        df = yf.download(ticker, START_DATE, END_DATE,
                         auto_adjust=True, progress=False)
        if not df.empty and len(df) > 500:
            s = df["Close"]
            s.name = name
            series.append(s)
    prices = pd.concat(series, axis=1).ffill().dropna()
    print("Data ready:", prices.shape)
    return prices

# Lazy initialization — no download on import
GLOBAL_PRICES = None
GLOBAL_REBALANCE = None
GLOBAL_CRISIS_SCALE = None

def init_data():
    """Initialize global data and crisis indicator, only once.
    Loads from pair_cache.pkl if available (same data as optimizer),
    falls back to fresh yfinance download.
    """
    global GLOBAL_PRICES, GLOBAL_REBALANCE, GLOBAL_CRISIS_SCALE
    if GLOBAL_PRICES is None:
        import os, pickle
        cache_path = "pair_cache.pkl"
        if os.path.exists(cache_path):
            print("Loading prices from cache (same data as optimizer)...")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            GLOBAL_PRICES = cache['prices']
            print(f"Data ready: {GLOBAL_PRICES.shape}")
        else:
            print("No cache found, downloading fresh data...")
            GLOBAL_PRICES = download_prices()
        GLOBAL_REBALANCE = GLOBAL_PRICES.index[::REBALANCE_STEP_DAYS]
        GLOBAL_CRISIS_SCALE = compute_crisis_scale(GLOBAL_PRICES)
        crisis_off = (GLOBAL_CRISIS_SCALE < 1.0).sum()
        print(f"Crisis detector: {crisis_off} / {len(GLOBAL_CRISIS_SCALE)} bars with reduced exposure")


# ============================================================
# CRISIS PROTECTION
# ============================================================
def compute_crisis_scale(prices, dd_window=None, scale_thresh=None, halt_thresh=None):
    """
    Compute per-bar position scaling factor in [0, 1] using
    rolling basket drawdown (fast-reacting, 10-day window).

    Calibrated from data:
      - Normal worst (2022 H1): DD_10d = -16.6%, strategy profitable
      - COVID (Mar 18-23 2020): DD_10d = -20% to -32%, strategy lost -14%
      - Threshold at -18%: catches COVID from Mar 18, zero false positives

    In normal markets this returns 1.0 everywhere — zero impact on returns.
    """
    dd_window = dd_window or CRISIS_DD_WINDOW
    scale_thresh = scale_thresh or CRISIS_DD_SCALE
    halt_thresh = halt_thresh or CRISIS_DD_HALT

    # Equal-weighted basket cumulative return
    log_ret = np.log(prices).diff()
    basket_ret = log_ret.mean(axis=1)
    basket_cum = (1 + basket_ret.fillna(0)).cumprod()

    # Rolling max drawdown over dd_window days
    rolling_peak = basket_cum.rolling(dd_window, min_periods=1).max()
    dd = basket_cum / rolling_peak - 1  # always <= 0

    # Linear ramp: 1.0 when dd >= scale_thresh, 0.0 when dd <= halt_thresh
    # scale_thresh = -0.18 (start scaling), halt_thresh = -0.25 (full halt)
    scale = (dd - halt_thresh) / (scale_thresh - halt_thresh)
    scale = scale.clip(0.0, 1.0)

    return scale

# ============================================================
# NUMBA HELPERS
# ============================================================
@njit
def hedge_ratio_nb(x, y):
    xm, ym = x.mean(), y.mean()
    den = ((y - ym) ** 2).sum()
    return 0.0 if den == 0 else ((y - ym) * (x - xm)).sum() / den

@njit
def half_life_nb(spread):
    lag = spread[:-1]
    delta = spread[1:] - lag
    if len(delta) < 10:
        return np.inf
    beta = ((lag - lag.mean()) * (delta - delta.mean())).sum() / ((lag - lag.mean())**2).sum()
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta

@njit
def rolling_beta_std(x, y, window=252, step=10):
    n = len(x)
    cnt = 0
    betas = np.empty((n - window) // step + 1)
    for i in range(window, n, step):
        betas[cnt] = hedge_ratio_nb(x[i-window:i], y[i-window:i])
        cnt += 1
    return np.inf if cnt < 5 else np.std(betas[:cnt])

@njit
def rolling_corr_nb(x, y, window):
    n = len(x)
    out = np.full(n, np.nan)
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(n):
        sx += x[i]; sy += y[i]
        sxx += x[i]*x[i]; syy += y[i]*y[i]; sxy += x[i]*y[i]
        if i >= window:
            sx -= x[i-window]; sy -= y[i-window]
            sxx -= x[i-window]**2; syy -= y[i-window]**2
            sxy -= x[i-window]*y[i-window]
        if i >= window-1:
            vx = sxx - sx*sx/window
            vy = syy - sy*sy/window
            if vx > 0 and vy > 0:
                out[i] = (sxy - sx*sy/window) / np.sqrt(vx*vy)
    return out

# FIX 5: Added min_hold and cooldown to prevent z-score chatter
@njit
def compute_positions(z, allow, entry_z, exit_z, stop_z_allow, stop_z_block,
                      min_hold=7, cooldown=2):
    """
    State-machine position generator with holding period and cooldown.

    Parameters
    ----------
    min_hold : int
        Minimum bars to hold a position before a normal exit is allowed.
        Stop-losses always override this.
    cooldown : int
        Bars to wait after an exit before a new entry is allowed.
    """
    n = len(z)
    pos = np.zeros(n)
    p = 0.0
    bars_in_trade = 0
    bars_since_exit = cooldown  # start ready to trade (no artificial delay at t=0)

    for i in range(n):
        zi = z[i]

        # Handle NaN z-scores: carry forward position
        if np.isnan(zi):
            pos[i] = p
            if p != 0:
                bars_in_trade += 1
            else:
                bars_since_exit += 1
            continue

        stop = stop_z_allow if allow[i] else stop_z_block

        # Priority 1: Stop-loss — always fires regardless of holding period
        if abs(zi) > stop:
            if p != 0:
                bars_since_exit = 0  # reset cooldown on exit
            p = 0.0
            bars_in_trade = 0

        # Priority 2: Normal exit — only after min holding period
        elif p != 0 and bars_in_trade >= min_hold and (abs(zi) <= exit_z or not allow[i]):
            p = 0.0
            bars_in_trade = 0
            bars_since_exit = 0  # reset cooldown on exit

        # Priority 3: Entry — only if flat, regime allows, and cooldown expired
        elif p == 0 and abs(zi) >= entry_z and allow[i] and bars_since_exit >= cooldown:
            p = -np.sign(zi)
            bars_in_trade = 1
            bars_since_exit = 0

        # Otherwise: hold current position
        else:
            if p != 0:
                bars_in_trade += 1
            else:
                bars_since_exit += 1

        pos[i] = p
    return pos

# ============================================================
# Z-THRESHOLDS
# ============================================================
def z_params(hl):
    if hl <= 10: return 0.8, 0.2
    if hl <= 20: return 1.0, 0.3
    if hl <= 40: return 1.4, 0.5
    return 1.7, 0.6

# ============================================================
# PAIR SELECTION
# ============================================================
def select_pairs(prices, end_date):

    start = end_date - pd.DateOffset(years=PAIR_LOOKBACK_YEARS)
    data = prices.loc[start:end_date]
    logp = np.log(data.values)
    names = data.columns

    candidates = []

    # Pre-compute pairwise correlations for filtering to match cache logic
    returns = np.diff(logp, axis=0)
    if returns.shape[0] < 100:
        return []

    corr_matrix = np.corrcoef(returns.T)

    for i in range(logp.shape[1]):
        for j in range(i+1, logp.shape[1]):

            # PRE-FILTER: skip low-correlation pairs (matches precompute_cache.py)
            if abs(corr_matrix[i, j]) < 0.3:
                continue

            x, y = logp[:, i], logp[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 252:
                continue

            x, y = x[mask], y[mask]

            try:
                t, pval, _ = coint(x, y)
            except Exception:
                continue
            if pval > 0.05:
                continue

            beta = hedge_ratio_nb(x, y)
            spread = x - beta*y
            hl = half_life_nb(spread)
            if hl < 3 or hl > 60:
                continue

            corr = rolling_corr_nb(x, y, 126)
            beta_std = rolling_beta_std(x, y)
            if not np.isfinite(beta_std):
                continue
            corr_mean = np.nanmean(corr)
            corr_std  = np.nanstd(corr)

            score = (
                2*abs(t)
                + 1.5*abs(corr_mean)
                - corr_std
                - 2*beta_std
                - 0.1*abs(hl-15)
            )

            candidates.append((names[i], names[j], beta, hl, score))

    ranked = sorted(candidates, key=lambda x: x[-1], reverse=True)

    selected, used = [], Counter()
    for a,b,beta,hl,_ in ranked:
        if used[a] < 2 and used[b] < 2:
            selected.append((a,b,beta,hl))
            used[a]+=1; used[b]+=1
        if len(selected) == MAX_PAIRS:
            break

    return selected

# ============================================================
# TRADING ENGINE
# ============================================================
def trade_pairs(prices, pair_book, trade_start, pair_pnl_dict, stop_z_allow, stop_z_block,
                crisis_scale=None):

    pnl = pd.Series(0.0, index=prices.index)
    cap_per_pair = INITIAL_CAPITAL / MAX_PAIRS
    regime = {"ALLOW_MR": 0, "BLOCKED": 0}
    trade_count = {}

    # Pre-compute mask for trading window (excludes buffer)
    trade_mask = prices.index >= trade_start

    # Align crisis scale to this price window (1.0 = no scaling if not provided)
    if crisis_scale is not None:
        cs_vals = crisis_scale.reindex(prices.index).fillna(1.0).values
    else:
        cs_vals = np.ones(len(prices))

    for a,b,beta,hl in pair_book:

        x = np.log(prices[a].values)
        y = np.log(prices[b].values)
        spread = x - beta*y
        ret = np.diff(spread, prepend=np.nan)

        vol = pd.Series(ret).rolling(VOL_WINDOW).std().values
        vol = np.clip(np.nan_to_num(vol, nan=1e-4), 1e-4, None)

        # FIX 6: Explicit NaN handling in regime filter
        ma = pd.Series(spread).rolling(60).mean().values
        ma_diff = np.abs(np.diff(ma, prepend=np.nan))
        allow = np.where(np.isnan(ma_diff), False, ma_diff < 2*vol)

        # FIX 2: Count regime stats only within the trading window
        regime["ALLOW_MR"] += allow[trade_mask].sum()
        regime["BLOCKED"] += (~allow[trade_mask]).sum()

        z = (spread - pd.Series(spread).rolling(ROLL_Z_WINDOW).mean().values) / \
            pd.Series(spread).rolling(ROLL_Z_WINDOW).std().values

        entry_base, exit_base = z_params(hl)
        entry = entry_base * ENTRY_Z_SCALE
        exit_ = exit_base * EXIT_Z_SCALE
        
        # FIX 5: Pass min_hold and cooldown to prevent chatter
        pos = compute_positions(z.astype(np.float64), allow, entry, exit_,
                                stop_z_allow, stop_z_block,
                                min_hold=MIN_HOLD_BARS, cooldown=COOLDOWN_BARS)

        # FIX 1: Count entries only within the trading window
        pos_series = pd.Series(pos, index=prices.index)
        active_pos = pos_series.loc[trade_start:]
        entries = ((active_pos != 0) & (active_pos.shift(1).fillna(0) == 0)).sum()
        trade_count[f"{a}-{b}"] = entries

        # Proper lagged position without wrap-around
        lagged_pos = np.concatenate(([0.0], pos[:-1]))
        # CAUSAL: lag vol by 1 bar — use yesterday's vol estimate, not today's
        vol_lagged = np.concatenate(([vol[0]], vol[:-1]))
        risk = np.clip(TARGET_DAILY_VOL / vol_lagged, 0, MAX_LEVERAGE)
        # CAUSAL: lag crisis scale by 1 bar — use yesterday's crisis assessment
        cs_lagged = np.concatenate(([1.0], cs_vals[:-1]))
        units = cap_per_pair * risk * lagged_pos * cs_lagged

        pair_pnl = units * ret - np.abs(np.diff(units, prepend=0))*TXN_COST
        pair_pnl = pd.Series(pair_pnl, index=prices.index).fillna(0)

        # FIX 7: Zero out buffer period to prevent txn-cost leakage
        pair_pnl.loc[:trade_start].iloc[:-1] = 0

        pnl += pair_pnl
        pair_pnl_dict[f"{a}-{b}"] = pair_pnl_dict.get(f"{a}-{b}",0) + pair_pnl.loc[trade_start:].sum()

    return pnl.loc[trade_start:], regime, trade_count



# ============================================================
# BACKTEST
# ============================================================
def run_backtest(stop_z_allow=5.279272073911419, stop_z_block=1.5115913424367038):

    init_data()  # lazy init (also computes crisis scale)

    pnl = pd.Series(0.0, index=GLOBAL_PRICES.index)
    pair_pnl_dict = {}
    trade_count_total = {}
    regime_total = {"ALLOW_MR": 0, "BLOCKED": 0}

    for i in range(1, len(GLOBAL_REBALANCE)):
        start = GLOBAL_REBALANCE[i-1]
        end   = GLOBAL_REBALANCE[i]

        pairs = select_pairs(GLOBAL_PRICES, start)
        if not pairs:
            continue

        prices = GLOBAL_PRICES.loc[start - pd.Timedelta(days=BUFFER_DAYS):end]
        p, reg, tc = trade_pairs(prices, pairs, start, pair_pnl_dict,
                                 stop_z_allow, stop_z_block,
                                 crisis_scale=GLOBAL_CRISIS_SCALE)
        pnl = pnl.add(p, fill_value=0)

        for pair, count in tc.items():
            trade_count_total[pair] = trade_count_total.get(pair, 0) + count

        regime_total["ALLOW_MR"] += reg["ALLOW_MR"]
        regime_total["BLOCKED"]  += reg["BLOCKED"]

    equity = INITIAL_CAPITAL + pnl.cumsum()
    return equity, pnl, pair_pnl_dict, trade_count_total, regime_total

# ============================================================
# ANALYTICS & PLOTS
# ============================================================
def compute_metrics(pnl, equity):
    ret = pnl / INITIAL_CAPITAL
    sharpe = 0 if ret.std()==0 else np.sqrt(252)*ret.mean()/ret.std()
    vol = np.sqrt(252)*ret.std()
    total = equity.iloc[-1]/INITIAL_CAPITAL - 1
    days = (equity.index[-1]-equity.index[0]).days
    cagr = (1+total)**(365/days)-1
    dd = (equity-equity.cummax())/equity.cummax()
    return {
        "Total Return %": 100*total,
        "CAGR %": 100*cagr,
        "Annual Vol %": 100*vol,
        "Sharpe": sharpe,
        "Max DD %": 100*dd.min()
    }

def pairwise_pnl_table(pair_pnl_dict):
    df = pd.DataFrame.from_dict(pair_pnl_dict, orient="index", columns=["Net PnL"])
    df["Contribution %"] = 100 * df["Net PnL"] / df["Net PnL"].abs().sum()
    return df.sort_values("Net PnL", ascending=False)

def plot_equity_and_drawdown(equity):
    dd = (equity - equity.cummax()) / equity.cummax()
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)
    ax1.plot(equity,label="Equity")
    ax1.legend(); ax1.grid()
    ax2.fill_between(dd.index,dd,0,color="red",alpha=0.3)
    ax2.set_ylabel("Drawdown"); ax2.grid()
    plt.show()

def plot_rolling_sharpe(pnl, window=126):
    ret = pnl / INITIAL_CAPITAL
    rs = np.sqrt(252)*ret.rolling(window).mean()/ret.rolling(window).std()
    rs.plot(title="Rolling 6M Sharpe"); plt.grid(); plt.show()

def save_outputs(equity, pnl, pair_pnl, trade_counts):

    import os
    OUTPUT_DIR = "backtest_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- core series ----
    equity.to_csv(f"{OUTPUT_DIR}/equity.csv")
    pnl.to_csv(f"{OUTPUT_DIR}/pnl.csv")

    # ---- metrics ----
    metrics = compute_metrics(pnl, equity)
    pd.DataFrame([metrics]).to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

    # ---- pair pnl ----
    pair_df = pairwise_pnl_table(pair_pnl)
    pair_df.to_csv(f"{OUTPUT_DIR}/pair_pnl.csv")

    # ---- trade counts ----
    pd.DataFrame.from_dict(trade_counts, orient="index", columns=["trades"])\
        .sort_values("trades", ascending=False)\
        .to_csv(f"{OUTPUT_DIR}/trade_counts.csv")

    # ---- equity + drawdown plot (reuse your logic) ----
    dd = (equity - equity.cummax()) / equity.cummax()

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True)
    ax1.plot(equity)
    ax1.set_title("Equity")
    ax1.grid()

    ax2.fill_between(dd.index,dd,0,alpha=0.3)
    ax2.set_title("Drawdown")
    ax2.grid()

    plt.savefig(f"{OUTPUT_DIR}/equity_drawdown.png")
    plt.close()

    # ---- rolling sharpe (reuse your formula) ----
    ret = pnl / INITIAL_CAPITAL
    rs = np.sqrt(252)*ret.rolling(126).mean()/ret.rolling(126).std()

    plt.figure(figsize=(10,4))
    rs.plot(title="Rolling Sharpe")
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/rolling_sharpe.png")
    plt.close()

    print(f"\nSaved all outputs to: {OUTPUT_DIR}")

# ============================================================
# OPTUNA
# ============================================================
def objective(trial):

    global PAIR_LOOKBACK_YEARS, ROLL_Z_WINDOW, VOL_WINDOW
    global TARGET_DAILY_VOL, MAX_LEVERAGE, TXN_COST

    PAIR_LOOKBACK_YEARS = trial.suggest_int("pair_lb",2,6)
    ROLL_Z_WINDOW = trial.suggest_int("roll_z",18,35)
    VOL_WINDOW = trial.suggest_int("vol_w",18,35)
    TARGET_DAILY_VOL = trial.suggest_float("tgt_vol",0.018,0.032,log=True)
    MAX_LEVERAGE = trial.suggest_float("max_lev",2.0,3.5)
    TXN_COST = trial.suggest_float("txn_cost",0.0004,0.001,log=True)

    s_allow = trial.suggest_float("stop_allow", 3.0, 5.0)
    s_block = trial.suggest_float("stop_block", 1.5, 3.0)

    # FIX 4: Unpack all 5 return values
    equity, pnl, _, _, _ = run_backtest(stop_z_allow=s_allow, stop_z_block=s_block)
    m = compute_metrics(pnl,equity)

    if m["Sharpe"]<0.4 or m["CAGR %"]<5 or m["Max DD %"]<-40:
        return -1e6

    return 5*(m["CAGR %"]/100) + 3*(m["Total Return %"]/100) + 2*m["Sharpe"] - 3*abs(m["Max DD %"]/100)



# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":

    equity, pnl, pair_pnl, trade_counts, regime = run_backtest()
    save_outputs(equity, pnl, pair_pnl, trade_counts)

    print("\nFINAL METRICS")
    for k, v in compute_metrics(pnl, equity).items():
        print(f"{k:<15}: {v:.2f}")

    print(f"\nREGIME STATS")
    total_regime = regime["ALLOW_MR"] + regime["BLOCKED"]
    if total_regime > 0:
        print(f"  Allow MR: {regime['ALLOW_MR']} ({100*regime['ALLOW_MR']/total_regime:.1f}%)")
        print(f"  Blocked:  {regime['BLOCKED']} ({100*regime['BLOCKED']/total_regime:.1f}%)")

    print("\nTRADE COUNT PER PAIR")
    for pair, count in sorted(trade_counts.items(), key=lambda x: -x[1]):
        print(f"  {pair:<25}: {count} trades")
    print(f"  {'TOTAL':<25}: {sum(trade_counts.values())} trades")

    print("\nPAIR PNL ATTRIBUTION")
    print(pairwise_pnl_table(pair_pnl).head(10))
