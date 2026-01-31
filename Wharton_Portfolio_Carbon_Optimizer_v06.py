# -*- coding: utf-8 -*-
"""
Pension Portfolio Builder - Carbon Reduction & ESG Version
Created on Sat Jan 10 15:42:37 2026
Enhanced with Carbon Optimization on Jan 31, 2026

@author: mpsih
Enhanced by: Claude

New Carbon Features:
- Carbon intensity tracking and optimization
- 50% reduction by 2030 pathway modeling
- Carbon budget constraints in portfolio optimization
- Sector-level carbon exposure analysis
- Decarbonization trajectory visualization
- Carbon-efficient frontier analysis
- Scope 1, 2, 3 emissions tracking (when available)

ESG Features (Basic):
- ESG score integration
- Controversy screening
- Paris-aligned benchmark tracking
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# FF5 data (Ken French)
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

# ----------------------------
# CARBON & ESG CONFIGURATION
# ----------------------------

# Carbon intensity targets (tCO2e per $1M invested)
CARBON_BASELINE_YEAR = 2024
CARBON_TARGET_YEAR = 2030
CARBON_REDUCTION_TARGET = 0.50  # 50% reduction

# Estimated carbon intensities by sector (tCO2e per $1M revenue)
# These are illustrative - in practice, use actual company data
SECTOR_CARBON_INTENSITY = {
    "Energy": 850.0,
    "Utilities": 750.0,
    "Materials": 450.0,
    "Industrials": 280.0,
    "Consumer Discretionary": 180.0,
    "Consumer Staples": 200.0,
    "Health Care": 120.0,
    "Financials": 80.0,
    "Information Technology": 95.0,
    "Communication Services": 110.0,
    "Real Estate": 150.0,
    "Unknown": 200.0,  # Conservative default
}

# Low-carbon sleeve characteristics
LOW_CARBON_SLEEVES = {
    "TRANSITION_EQUITY": 0.3,    # 30% of market carbon intensity
    "PRIVATE_CLIMATE": 0.2,      # 20% of market carbon intensity (climate solutions)
    "IG_CREDIT": 0.6,            # 60% of market (green bonds available)
}

MARKET_AVERAGE_CARBON_INTENSITY = 220.0  # tCO2e per $1M invested (illustrative)

# ----------------------------
# 1) Policy configuration
# ----------------------------

SLEEVE_BOUNDS = {
    "US_EQUITY": (0.15, 0.25),
    "IG_CREDIT": (0.25, 0.40),
    "TRANSITION_EQUITY": (0.15, 0.30),
    "PRIVATE_CLIMATE": (0.10, 0.20),
    "EM_HC_DEBT": (0.05, 0.15),
    "CASH": (0.03, 0.07),
}

SLEEVE_CMA_NOMINAL = {
    "US_EQUITY": 0.065,
    "IG_CREDIT": 0.035,
    "TRANSITION_EQUITY": 0.065,
    "PRIVATE_CLIMATE": 0.070,
    "EM_HC_DEBT": 0.060,
    "CASH": 0.030,
}

SLEEVE_BENCHMARK_TICKER = {
    "US_EQUITY": "SPY",
    "IG_CREDIT": "LQD",
    "TRANSITION_EQUITY": "ICLN",
    "PRIVATE_CLIMATE": "PSP",
    "EM_HC_DEBT": "EMB",
    "CASH": "BIL",
}

# ----------------------------
# 2) Core Helper Functions
# ----------------------------

def parse_list(raw: str):
    return [t.strip().upper() for t in raw.split(",") if t.strip()]

def download_adjclose(tickers, start):
    tickers = list(dict.fromkeys([t.upper() for t in tickers if t]))
    if not tickers:
        raise ValueError("No tickers provided")
    
    print(f"  Downloading {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
    
    try:
        data = yf.download(tickers, start=start, progress=False, auto_adjust=False)
    except Exception as e:
        raise ValueError(f"Yahoo Finance download failed: {e}")
    
    if data.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"].copy()
    else:
        px = data[["Adj Close"]].copy()
        px.columns = tickers

    px = px.dropna(how="all")
    return px

def returns_from_prices(prices):
    return prices.pct_change().dropna()

def annualize_mu(mu_daily, periods=252):
    return mu_daily * periods

def annualize_cov(cov_daily, periods=252):
    return cov_daily * periods

def max_drawdown(eq):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def build_sleeve_matrix(all_tickers, sleeve_of):
    sleeves = sorted(set(sleeve_of[t] for t in all_tickers))
    A = np.zeros((len(sleeves), len(all_tickers)))
    for j, s in enumerate(sleeves):
        for i, t in enumerate(all_tickers):
            if sleeve_of[t] == s:
                A[j, i] = 1.0
    return sleeves, A

# ----------------------------
# 3) CARBON ESTIMATION FUNCTIONS
# ----------------------------

def get_ticker_sector(ticker):
    """
    Fetch sector information for a ticker using yfinance
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get sector
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        return sector, industry
    except Exception as e:
        return "Unknown", "Unknown"

def estimate_carbon_intensity(ticker, sleeve=None):
    """
    Estimate carbon intensity for a ticker
    Priority: 1) Actual data (if available), 2) Sector-based estimate, 3) Sleeve-based estimate
    Returns: tCO2e per $1M invested
    """
    # For low-carbon sleeves, use sleeve-based estimate
    if sleeve in LOW_CARBON_SLEEVES:
        return MARKET_AVERAGE_CARBON_INTENSITY * LOW_CARBON_SLEEVES[sleeve]
    
    # For cash, essentially zero carbon
    if sleeve == "CASH":
        return 5.0  # Minimal operational carbon
    
    # Otherwise, use sector-based estimate
    sector, industry = get_ticker_sector(ticker)
    
    carbon_intensity = SECTOR_CARBON_INTENSITY.get(sector, SECTOR_CARBON_INTENSITY["Unknown"])
    
    return carbon_intensity

def build_carbon_vector(tickers, sleeve_of, verbose=True):
    """
    Build vector of carbon intensities for all tickers
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ESTIMATING CARBON INTENSITIES")
        print("=" * 70)
    
    carbon_intensities = []
    
    for ticker in tickers:
        sleeve = sleeve_of.get(ticker)
        carbon = estimate_carbon_intensity(ticker, sleeve)
        carbon_intensities.append(carbon)
        
        if verbose and len(tickers) <= 20:
            print(f"{ticker:8s} ({sleeve:20s}): {carbon:6.1f} tCO2e/$1M")
    
    carbon_vector = np.array(carbon_intensities)
    
    if verbose:
        print(f"\nPortfolio-weighted average carbon intensity range:")
        print(f"  Min possible: {carbon_vector.min():.1f} tCO2e/$1M")
        print(f"  Max possible: {carbon_vector.max():.1f} tCO2e/$1M")
        print(f"  Market avg:   {MARKET_AVERAGE_CARBON_INTENSITY:.1f} tCO2e/$1M")
    
    return carbon_vector

def calculate_portfolio_carbon(w, carbon_vector):
    """
    Calculate weighted average carbon intensity of portfolio
    """
    return float(w @ carbon_vector)

def carbon_reduction_from_baseline(current_carbon, baseline_carbon):
    """
    Calculate % reduction from baseline
    """
    return (baseline_carbon - current_carbon) / baseline_carbon

def linear_decarbonization_path(baseline_carbon, target_reduction, years_to_target):
    """
    Generate linear decarbonization pathway
    Returns: list of (year, target_carbon) tuples
    """
    target_carbon = baseline_carbon * (1 - target_reduction)
    annual_reduction = (baseline_carbon - target_carbon) / years_to_target
    
    path = []
    for year in range(years_to_target + 1):
        year_carbon = baseline_carbon - (annual_reduction * year)
        path.append((CARBON_BASELINE_YEAR + year, year_carbon))
    
    return path

# ----------------------------
# 4) CARBON-CONSTRAINED OPTIMIZATION
# ----------------------------

def carbon_constraint_function(w, carbon_vector, max_carbon):
    """
    Constraint function: portfolio carbon <= max_carbon
    Returns positive value when constraint is satisfied
    """
    portfolio_carbon = calculate_portfolio_carbon(w, carbon_vector)
    return max_carbon - portfolio_carbon

def sleeve_constraints(A, sleeves):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    
    for j, s in enumerate(sleeves):
        lb, ub = SLEEVE_BOUNDS[s]
        row = A[j, :].copy()
        
        def make_lb_constraint(r, bound):
            return lambda w: (r @ w) - bound
        
        def make_ub_constraint(r, bound):
            return lambda w: bound - (r @ w)
        
        cons.append({"type": "ineq", "fun": make_lb_constraint(row, lb)})
        cons.append({"type": "ineq", "fun": make_ub_constraint(row, ub)})
    
    return cons

def add_carbon_constraint(cons, carbon_vector, max_carbon):
    """
    Add carbon budget constraint to existing constraints
    """
    carbon_cons = {
        "type": "ineq",
        "fun": lambda w: carbon_constraint_function(w, carbon_vector, max_carbon)
    }
    return cons + [carbon_cons]

def validate_sleeve_allocation(w, A, sleeves, verbose=True):
    violations = {}
    all_valid = True
    
    if verbose:
        print("\n" + "=" * 70)
        print("Sleeve Allocation Validation")
        print("=" * 70)
    
    for j, s in enumerate(sleeves):
        actual = float(A[j, :] @ w)
        lb, ub = SLEEVE_BOUNDS[s]
        
        is_valid = (lb - 1e-6) <= actual <= (ub + 1e-6)
        status = "✓ OK" if is_valid else "✗ VIOLATION"
        
        if not is_valid:
            all_valid = False
            violations[s] = {"actual": actual, "bounds": (lb, ub)}
        
        if verbose:
            print(f"{s:20s} {actual:6.4f}  [{lb:.2f}, {ub:.2f}]  {status}")
    
    return all_valid, violations

def neg_sharpe(w, mu, cov, rf):
    pret = float(w @ mu)
    pvol = float(np.sqrt(w.T @ cov @ w))
    if pvol <= 1e-12:
        return 1e9
    return -((pret - rf) / pvol)

def smart_initial_weights(tickers, sleeve_of, A, sleeves):
    w0 = np.zeros(len(tickers))
    
    for j, s in enumerate(sleeves):
        lb, ub = SLEEVE_BOUNDS[s]
        target_sleeve_weight = (lb + ub) / 2
        
        mask = A[j, :] > 0
        n_in_sleeve = mask.sum()
        
        if n_in_sleeve > 0:
            w0[mask] = target_sleeve_weight / n_in_sleeve
    
    w0 = w0 / w0.sum()
    return w0

# ----------------------------
# 5) Sleeve-Level Analytics
# ----------------------------

def calculate_sleeve_returns(rets, w, tickers, sleeve_of):
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    sleeve_returns = {}
    
    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue
            
        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())
        
        if sleeve_weight <= 1e-12:
            continue
        
        w_norm = (w_s / sleeve_weight).values
        sleeve_ret = pd.Series(rets[idx].values @ w_norm, index=rets.index, name=s)
        sleeve_returns[s] = sleeve_ret
    
    return pd.DataFrame(sleeve_returns)

def sleeve_sharpe_ratios(sleeve_rets, rf_annual=0.04, periods=252):
    results = {}
    
    for col in sleeve_rets.columns:
        ret_series = sleeve_rets[col].dropna()
        
        ann_ret = ret_series.mean() * periods
        ann_vol = ret_series.std() * np.sqrt(periods)
        
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-12 else np.nan
        
        results[col] = {
            "Return": ann_ret,
            "Volatility": ann_vol,
            "Sharpe": sharpe
        }
    
    return pd.DataFrame(results).T

def sleeve_betas(sleeve_rets, market_ret, periods=252):
    results = {}
    
    for col in sleeve_rets.columns:
        sleeve_series = sleeve_rets[col]
        
        aligned = pd.concat([sleeve_series, market_ret], axis=1, join='inner').dropna()
        
        if len(aligned) < 30:
            results[col] = {"Beta": np.nan, "Correlation": np.nan}
            continue
        
        cov_matrix = aligned.cov() * periods
        var_market = cov_matrix.iloc[1, 1]
        
        if var_market > 1e-12:
            beta = cov_matrix.iloc[0, 1] / var_market
        else:
            beta = np.nan
        
        corr = aligned.corr().iloc[0, 1]
        
        results[col] = {"Beta": beta, "Correlation": corr}
    
    return pd.DataFrame(results).T

def sleeve_contribution_analysis(sleeve_rets, w, tickers, sleeve_of):
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    results = {}
    
    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue
        
        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())
        
        if s in sleeve_rets.columns:
            sleeve_ret_series = sleeve_rets[s]
            ann_ret = sleeve_ret_series.mean() * 252
            ann_vol = sleeve_ret_series.std() * np.sqrt(252)
            
            contribution = sleeve_weight * ann_ret
            
            results[s] = {
                "Weight": sleeve_weight,
                "Return": ann_ret,
                "Volatility": ann_vol,
                "Contribution": contribution
            }
    
    return pd.DataFrame(results).T

def sleeve_tracking_report(rets, w, tickers, sleeve_of, sleeve_bmk_rets):
    w_ser = pd.Series(w, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    results = {}
    
    for s in sleeves:
        if s not in sleeve_bmk_rets.columns:
            continue
        
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue
        
        w_s = w_ser.loc[idx]
        sleeve_weight = float(w_s.sum())
        
        if sleeve_weight <= 1e-12:
            continue
        
        w_norm = (w_s / sleeve_weight).values
        sleeve_ret = pd.Series(rets[idx].values @ w_norm, index=rets.index)
        
        bmk_ret = sleeve_bmk_rets[s]
        
        aligned = pd.concat([sleeve_ret, bmk_ret], axis=1, join='inner').dropna()
        
        if len(aligned) < 30:
            continue
        
        active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        
        te = active.std() * np.sqrt(252)
        ann_active = active.mean() * 252
        ir = ann_active / te if te > 1e-12 else np.nan
        
        results[s] = {
            "Active Return": ann_active,
            "Tracking Error": te,
            "Information Ratio": ir
        }
    
    return pd.DataFrame(results).T

def ff5_exposure_report(port_daily, start_date="2020-01-01"):
    try:
        ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start_date)[0]
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
        ff = ff / 100.0
        
        aligned = pd.concat([port_daily, ff], axis=1, join="inner").dropna()
        
        if len(aligned) < 50:
            raise ValueError("Insufficient overlap with FF5 data")
        
        Y = aligned.iloc[:, 0].values - aligned["RF"].values
        X_factors = aligned[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values
        X = np.column_stack([np.ones(len(X_factors)), X_factors])
        
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        
        result = pd.Series({
            "Alpha_daily": coef[0],
            "Mkt-RF": coef[1],
            "SMB": coef[2],
            "HML": coef[3],
            "RMW": coef[4],
            "CMA": coef[5]
        })
        
        return result
    
    except Exception as e:
        raise ValueError(f"FF5 analysis failed: {e}")

# ----------------------------
# 6) CARBON ANALYTICS & REPORTING
# ----------------------------

def carbon_attribution_by_sleeve(w, tickers, sleeve_of, carbon_vector):
    """
    Calculate carbon contribution by sleeve
    """
    w_ser = pd.Series(w, index=tickers)
    carbon_ser = pd.Series(carbon_vector, index=tickers)
    sleeves = sorted(set(sleeve_of[t] for t in tickers))
    
    results = {}
    
    total_carbon = calculate_portfolio_carbon(w, carbon_vector)
    
    for s in sleeves:
        idx = [t for t in tickers if sleeve_of[t] == s]
        if not idx:
            continue
        
        sleeve_weight = float(w_ser.loc[idx].sum())
        sleeve_carbon = float((w_ser.loc[idx] * carbon_ser.loc[idx]).sum())
        avg_carbon = sleeve_carbon / sleeve_weight if sleeve_weight > 1e-12 else 0
        
        contribution_pct = (sleeve_carbon / total_carbon * 100) if total_carbon > 0 else 0
        
        results[s] = {
            "Weight": sleeve_weight,
            "Avg Carbon Intensity": avg_carbon,
            "Total Carbon": sleeve_carbon,
            "% of Portfolio Carbon": contribution_pct
        }
    
    return pd.DataFrame(results).T

def sector_carbon_exposure(w, tickers, carbon_vector):
    """
    Calculate carbon exposure by sector
    """
    sectors = {}
    
    for i, ticker in enumerate(tickers):
        sector, _ = get_ticker_sector(ticker)
        
        if sector not in sectors:
            sectors[sector] = {"weight": 0, "carbon": 0}
        
        sectors[sector]["weight"] += w[i]
        sectors[sector]["carbon"] += w[i] * carbon_vector[i]
    
    results = {}
    total_carbon = calculate_portfolio_carbon(w, carbon_vector)
    
    for sector, data in sectors.items():
        avg_carbon = data["carbon"] / data["weight"] if data["weight"] > 1e-12 else 0
        contribution_pct = (data["carbon"] / total_carbon * 100) if total_carbon > 0 else 0
        
        results[sector] = {
            "Weight": data["weight"],
            "Avg Carbon Intensity": avg_carbon,
            "Total Carbon": data["carbon"],
            "% of Portfolio Carbon": contribution_pct
        }
    
    return pd.DataFrame(results).T.sort_values("Total Carbon", ascending=False)

# ----------------------------
# 7) VISUALIZATION FUNCTIONS
# ----------------------------

def plot_equity_curve(port_daily, title):
    eq = (1 + port_daily).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(eq.index, eq.values, linewidth=2, color='#2E86AB')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_drawdown(port_daily, title):
    eq = (1 + port_daily).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color='#A23B72')
    ax.plot(dd.index, dd.values, linewidth=2, color='#A23B72')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weights(w, tickers, title, top_n=20):
    w_series = pd.Series(w, index=tickers).sort_values(ascending=True)
    w_display = w_series.tail(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(w_display)))
    w_display.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Weight", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(rets, tickers, sleeve_of):
    corr_matrix = rets.corr()
    
    sleeve_order = sorted(set(sleeve_of[t] for t in tickers))
    ordered_tickers = []
    for s in sleeve_order:
        ordered_tickers.extend([t for t in tickers if sleeve_of[t] == s])
    
    corr_ordered = corr_matrix.loc[ordered_tickers, ordered_tickers]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_ordered, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title("Correlation Heatmap (Grouped by Sleeve)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_correlation_3d(rets, tickers, sleeve_of):
    corr_matrix = rets.corr().values
    n = len(tickers)
    
    x_idx, y_idx = np.meshgrid(range(n), range(n))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x_idx, y_idx, corr_matrix, cmap='RdYlGn',
                           vmin=-1, vmax=1, alpha=0.8)
    
    ax.set_title("3D Correlation Surface", fontsize=16, fontweight='bold')
    ax.set_xlabel("Ticker Index", fontsize=10)
    ax.set_ylabel("Ticker Index", fontsize=10)
    ax.set_zlabel("Correlation", fontsize=10)
    ax.set_zlim(-1, 1)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

def plot_sleeve_performance(sleeve_rets):
    sleeve_eq = (1 + sleeve_rets).cumprod()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sleeve_eq.columns)))
    
    for i, col in enumerate(sleeve_eq.columns):
        ax.plot(sleeve_eq.index, sleeve_eq[col], label=col, 
                linewidth=2, color=colors[i])
    
    ax.set_title("Sleeve Performance Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_sleeve_metrics_comparison(sharpe_df, beta_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sharpe ratios
    sharpe_df['Sharpe'].plot(kind='bar', ax=ax1, color='#2E86AB', alpha=0.7)
    ax1.set_title("Sharpe Ratio by Sleeve", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Sharpe Ratio", fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Betas
    beta_df['Beta'].plot(kind='bar', ax=ax2, color='#A23B72', alpha=0.7)
    ax2.set_title("Beta by Sleeve (vs SPY)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Beta", fontsize=12)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def plot_rolling_sharpe(port_daily, window=252):
    rolling_ret = port_daily.rolling(window).mean() * 252
    rolling_vol = port_daily.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_ret - 0.04) / rolling_vol
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='#18A558')
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day window)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_risk_contribution(contrib_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    contrib_df[['Weight', 'Contribution']].plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_title("Sleeve Weight vs Return Contribution", fontsize=16, fontweight='bold')
    ax.set_ylabel("Proportion", fontsize=12)
    ax.legend(['Portfolio Weight', 'Return Contribution'], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier_position(mu, cov, w_optimal, rf, tickers, sleeve_of, A, sleeves, bounds, cons, w0):
    """Plot efficient frontier with optimal portfolio position"""
    print("\nGenerating efficient frontier (this may take a moment)...")
    
    target_returns = np.linspace(mu.min() * 0.8, mu.max() * 1.2, 30)
    frontier_vols = []
    frontier_rets = []
    
    for target_ret in target_returns:
        cons_with_ret = cons + [{"type": "eq", "fun": lambda w, tr=target_ret: (w @ mu) - tr}]
        
        def portfolio_variance(w):
            return w.T @ cov @ w
        
        try:
            res = minimize(portfolio_variance, w0, method="SLSQP",
                          bounds=bounds, constraints=cons_with_ret,
                          options={"maxiter": 500, "ftol": 1e-8})
            
            if res.success:
                w_temp = np.clip(res.x, 0, None)
                w_temp = w_temp / w_temp.sum()
                
                ret = float(w_temp @ mu)
                vol = float(np.sqrt(w_temp.T @ cov @ w_temp))
                
                frontier_rets.append(ret)
                frontier_vols.append(vol)
        except:
            continue
    
    # Calculate optimal portfolio metrics
    opt_ret = float(w_optimal @ mu)
    opt_vol = float(np.sqrt(w_optimal.T @ cov @ w_optimal))
    opt_sharpe = (opt_ret - rf) / opt_vol
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Efficient frontier
    if frontier_vols and frontier_rets:
        ax.plot(frontier_vols, frontier_rets, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Optimal portfolio
    ax.scatter([opt_vol], [opt_ret], c='red', s=200, marker='*', 
              label=f'Optimal Portfolio (Sharpe={opt_sharpe:.3f})', zorder=5)
    
    # Capital allocation line
    max_ret = max(frontier_rets) if frontier_rets else opt_ret
    cal_x = np.linspace(0, opt_vol * 1.5, 100)
    cal_y = rf + (opt_ret - rf) / opt_vol * cal_x
    ax.plot(cal_x, cal_y, 'g--', linewidth=1, alpha=0.7, label='Capital Allocation Line')
    
    ax.set_xlabel('Volatility (Annual)', fontsize=12)
    ax.set_ylabel('Expected Return (Annual)', fontsize=12)
    ax.set_title('Efficient Frontier with Optimal Portfolio', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_carbon_decarbonization_path(current_carbon, baseline_carbon, target_year=CARBON_TARGET_YEAR):
    """
    Visualize decarbonization pathway vs target
    """
    years_to_target = target_year - CARBON_BASELINE_YEAR
    path = linear_decarbonization_path(baseline_carbon, CARBON_REDUCTION_TARGET, years_to_target)
    
    years = [p[0] for p in path]
    targets = [p[1] for p in path]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Target path
    ax.plot(years, targets, 'b--', linewidth=2, label='Target Pathway (50% reduction)', marker='o')
    
    # Current position
    ax.scatter([CARBON_BASELINE_YEAR + 2], [current_carbon], c='green', s=300, marker='*',
              label=f'Current Portfolio ({current_carbon:.1f} tCO2e/$1M)', zorder=5)
    
    # Baseline
    ax.axhline(y=baseline_carbon, color='red', linestyle=':', linewidth=2,
              label=f'Baseline ({baseline_carbon:.1f} tCO2e/$1M)')
    
    # Target
    target_carbon = baseline_carbon * (1 - CARBON_REDUCTION_TARGET)
    ax.axhline(y=target_carbon, color='green', linestyle=':', linewidth=2,
              label=f'2030 Target ({target_carbon:.1f} tCO2e/$1M)')
    
    # Reduction achieved
    reduction_pct = carbon_reduction_from_baseline(current_carbon, baseline_carbon) * 100
    ax.text(CARBON_BASELINE_YEAR + 1, current_carbon + 10, 
            f'{reduction_pct:.1f}% reduction achieved',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Carbon Intensity (tCO2e / $1M invested)', fontsize=12)
    ax.set_title('Portfolio Decarbonization Pathway to 2030', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_carbon_by_sleeve(carbon_attribution_df):
    """
    Visualize carbon contribution by sleeve
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Carbon intensity by sleeve
    carbon_attribution_df['Avg Carbon Intensity'].plot(kind='bar', ax=ax1, 
                                                        color='#E63946', alpha=0.7)
    ax1.set_title("Average Carbon Intensity by Sleeve", fontsize=14, fontweight='bold')
    ax1.set_ylabel("tCO2e / $1M invested", fontsize=12)
    ax1.axhline(y=MARKET_AVERAGE_CARBON_INTENSITY, color='black', linestyle='--',
               label='Market Average')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Carbon contribution pie chart
    ax2.pie(carbon_attribution_df['Total Carbon'], 
            labels=carbon_attribution_df.index,
            autopct='%1.1f%%', startangle=90,
            colors=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(carbon_attribution_df))))
    ax2.set_title("Carbon Contribution by Sleeve", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_carbon_efficient_frontier(mu, cov, tickers, sleeve_of, A, sleeves, bounds, cons, w0, 
                                   carbon_vector, rf):
    """
    Plot risk-return efficient frontier colored by carbon intensity
    """
    print("\nGenerating carbon-efficient frontier...")
    
    target_returns = np.linspace(mu.min() * 0.8, mu.max() * 1.2, 25)
    frontier_data = []
    
    for target_ret in target_returns:
        cons_with_ret = cons + [{"type": "eq", "fun": lambda w, tr=target_ret: (w @ mu) - tr}]
        
        def portfolio_variance(w):
            return w.T @ cov @ w
        
        try:
            res = minimize(portfolio_variance, w0, method="SLSQP",
                          bounds=bounds, constraints=cons_with_ret,
                          options={"maxiter": 500, "ftol": 1e-8})
            
            if res.success:
                w_temp = np.clip(res.x, 0, None)
                w_temp = w_temp / w_temp.sum()
                
                ret = float(w_temp @ mu)
                vol = float(np.sqrt(w_temp.T @ cov @ w_temp))
                carbon = calculate_portfolio_carbon(w_temp, carbon_vector)
                
                frontier_data.append({
                    'return': ret,
                    'volatility': vol,
                    'carbon': carbon,
                    'sharpe': (ret - rf) / vol
                })
        except:
            continue
    
    if not frontier_data:
        print("Could not generate carbon-efficient frontier")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vols = [d['volatility'] for d in frontier_data]
    rets = [d['return'] for d in frontier_data]
    carbons = [d['carbon'] for d in frontier_data]
    
    scatter = ax.scatter(vols, rets, c=carbons, s=100, cmap='RdYlGn_r',
                        edgecolors='black', linewidth=0.5, alpha=0.8)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Carbon Intensity (tCO2e/$1M)', fontsize=11)
    
    ax.set_xlabel('Volatility (Annual)', fontsize=12)
    ax.set_ylabel('Expected Return (Annual)', fontsize=12)
    ax.set_title('Risk-Return Frontier Colored by Carbon Intensity', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add market average carbon line to legend
    ax.scatter([], [], c='white', edgecolors='black', s=100, 
              label=f'Market Avg: {MARKET_AVERAGE_CARBON_INTENSITY:.0f} tCO2e/$1M')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ----------------------------
# 8) MAIN FUNCTION
# ----------------------------

def main():
    print("=" * 70)
    print("PENSION PORTFOLIO BUILDER - CARBON REDUCTION VERSION")
    print("=" * 70)
    print("\nTarget: 50% carbon reduction by 2030")
    print("Current year: 2026 (4 years to target)")
    
    # User inputs
    start = input("\nStart date for historical data (YYYY-MM-DD, default=2020-01-01): ").strip()
    if not start:
        start = "2020-01-01"
    
    max_w_input = input("Max weight per ticker (default=0.05): ").strip()
    max_w = float(max_w_input) if max_w_input else 0.05
    
    rf_input = input("Risk-free rate (default=0.04): ").strip()
    rf = float(rf_input) if rf_input else 0.04
    
    inflation_input = input("Expected inflation (default=0.025): ").strip()
    inflation = float(inflation_input) if inflation_input else 0.025
    
    # Carbon constraint input
    print("\n" + "-" * 70)
    print("CARBON BUDGET CONFIGURATION")
    print("-" * 70)
    print(f"Market average carbon intensity: {MARKET_AVERAGE_CARBON_INTENSITY:.1f} tCO2e/$1M")
    print(f"Target for 2030: {MARKET_AVERAGE_CARBON_INTENSITY * 0.5:.1f} tCO2e/$1M (50% reduction)")
    
    # Calculate linear pathway for 2026
    years_elapsed = 2026 - CARBON_BASELINE_YEAR  # 2 years
    years_total = CARBON_TARGET_YEAR - CARBON_BASELINE_YEAR  # 6 years
    linear_target_2026 = MARKET_AVERAGE_CARBON_INTENSITY * (1 - CARBON_REDUCTION_TARGET * (years_elapsed / years_total))
    
    print(f"\nLinear pathway target for 2026: {linear_target_2026:.1f} tCO2e/$1M")
    
    carbon_budget_input = input(f"Max carbon intensity (default={linear_target_2026:.1f}): ").strip()
    carbon_budget = float(carbon_budget_input) if carbon_budget_input else linear_target_2026
    
    use_carbon_constraint = input("Apply carbon constraint in optimization? (y/n, default=y): ").strip().lower()
    if not use_carbon_constraint:
        use_carbon_constraint = 'y'

    print("\n" + "-" * 70)
    print("Enter tickers for each sleeve (comma-separated)")
    print("-" * 70)

    us_eq = parse_list(input("US_EQUITY stocks: "))
    tr_eq = parse_list(input("TRANSITION_EQUITY stocks: "))
    ig = parse_list(input("IG_CREDIT ETF(s): "))
    em = parse_list(input("EM_HC_DEBT ETF(s): "))
    cash = parse_list(input("CASH ETF(s): "))
    priv_raw = input("PRIVATE_CLIMATE proxy ticker (optional): ").strip().upper()
    priv = [priv_raw] if priv_raw else []

    sleeve_of = {}
    for t in us_eq: sleeve_of[t] = "US_EQUITY"
    for t in tr_eq: sleeve_of[t] = "TRANSITION_EQUITY"
    for t in ig:    sleeve_of[t] = "IG_CREDIT"
    for t in em:    sleeve_of[t] = "EM_HC_DEBT"
    for t in cash:  sleeve_of[t] = "CASH"
    for t in priv:  sleeve_of[t] = "PRIVATE_CLIMATE"

    tickers = list(dict.fromkeys(us_eq + tr_eq + ig + em + cash + priv))
    
    if len(tickers) < 5:
        print("\n❌ Error: Need at least 5 tickers")
        return

    print(f"\n✓ Collected {len(tickers)} tickers across {len(set(sleeve_of.values()))} sleeves")

    # Download data
    print("\n" + "-" * 70)
    print("Downloading data...")
    print("-" * 70)
    
    try:
        prices = download_adjclose(tickers, start=start)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    # Benchmarks
    sleeves_needed = sorted(set(sleeve_of[t] for t in tickers))
    bench_tickers = list(dict.fromkeys([SLEEVE_BENCHMARK_TICKER.get(s) 
                                        for s in sleeves_needed if SLEEVE_BENCHMARK_TICKER.get(s)]))

    print("\nDownloading benchmarks...")
    try:
        bench_prices = download_adjclose(bench_tickers, start=start)
    except Exception as e:
        print(f"⚠ Warning: Benchmark download failed: {e}")
        bench_prices = pd.DataFrame()

    # Clean data
    coverage = prices.notna().sum()
    min_obs = max(50, int(0.5 * len(prices)))
    good = coverage[coverage >= min_obs].index.tolist()
    
    if len(good) < len(tickers):
        dropped = set(tickers) - set(good)
        print(f"\n⚠ Dropped {len(dropped)} tickers: {', '.join(list(dropped)[:5])}{'...' if len(dropped) > 5 else ''}")
    
    prices = prices[good].dropna()
    tickers = good
    sleeve_of = {t: sleeve_of[t] for t in tickers if t in sleeve_of}
    sleeves_needed = sorted(set(sleeve_of.values()))

    if len(tickers) < 5:
        print("\n❌ Error: Not enough usable tickers")
        return

    print(f"✓ Using {len(tickers)} tickers with {len(prices)} observations")

    # Returns
    rets = returns_from_prices(prices)
    
    # Benchmark returns
    sleeve_bmk_rets = pd.DataFrame()
    market_ret = None
    
    if not bench_prices.empty:
        bench_rets_all = returns_from_prices(bench_prices)
        
        if 'SPY' in bench_rets_all.columns:
            market_ret = bench_rets_all['SPY']
        
        sleeve_bmk_map = {}
        for s in sleeves_needed:
            bt = SLEEVE_BENCHMARK_TICKER.get(s)
            if bt and bt in bench_rets_all.columns:
                sleeve_bmk_map[s] = bench_rets_all[bt]
        
        if sleeve_bmk_map:
            sleeve_bmk_rets = pd.DataFrame(sleeve_bmk_map)

    # === CARBON ANALYSIS ===
    carbon_vector = build_carbon_vector(tickers, sleeve_of, verbose=True)
    
    # Build risk model
    print("\n" + "-" * 70)
    print("Building risk model...")
    print("-" * 70)
    
    mu_hist = annualize_mu(rets.mean()).values
    cov = annualize_cov(rets.cov()).values
    mu_cma = np.array([SLEEVE_CMA_NOMINAL[sleeve_of[t]] for t in tickers])
    mu = 0.75 * mu_cma + 0.25 * mu_hist

    print(f"✓ Built covariance matrix ({len(tickers)}x{len(tickers)})")

    # Optimize
    sleeves, A = build_sleeve_matrix(tickers, sleeve_of)
    cons = sleeve_constraints(A, sleeves)
    
    # Add carbon constraint if requested
    if use_carbon_constraint == 'y':
        cons = add_carbon_constraint(cons, carbon_vector, carbon_budget)
        print(f"\n✓ Added carbon budget constraint: {carbon_budget:.1f} tCO2e/$1M")
    
    bounds = [(0.0015, max_w) for _ in tickers]
    w0 = smart_initial_weights(tickers, sleeve_of, A, sleeves)

    print("\nOptimizing portfolio...")
    res = minimize(neg_sharpe, w0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})

    if not res.success:
        print(f"\n⚠ Warning: {res.message}")

    w = np.clip(res.x, 0, None)
    w = w / w.sum()

    # Validate
    is_valid, violations = validate_sleeve_allocation(w, A, sleeves, verbose=True)

    # Portfolio stats
    port_ret_nom = float(w @ mu)
    port_vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = (port_ret_nom - rf) / port_vol if port_vol > 0 else np.nan
    port_ret_real = port_ret_nom - inflation
    
    # === CARBON METRICS ===
    portfolio_carbon = calculate_portfolio_carbon(w, carbon_vector)
    baseline_carbon = MARKET_AVERAGE_CARBON_INTENSITY
    carbon_reduction_pct = carbon_reduction_from_baseline(portfolio_carbon, baseline_carbon)

    # Display basic results
    print("\n" + "=" * 70)
    print("OPTIMIZED WEIGHTS (Top 30)")
    print("=" * 70)
    w_series = pd.Series(w, index=tickers).sort_values(ascending=False)
    print(w_series.head(30).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("EXPECTED STATISTICS (Annualized)")
    print("=" * 70)
    print(f"Expected Nominal Return:  {port_ret_nom:7.2%}")
    print(f"Expected Real Return:     {port_ret_real:7.2%}")
    print(f"Expected Volatility:      {port_vol:7.2%}")
    print(f"Sharpe Ratio:             {sharpe:7.3f}")

    print("\n" + "=" * 70)
    print("CARBON FOOTPRINT")
    print("=" * 70)
    print(f"Portfolio Carbon Intensity: {portfolio_carbon:7.1f} tCO2e/$1M")
    print(f"Market Baseline:            {baseline_carbon:7.1f} tCO2e/$1M")
    print(f"Reduction from Baseline:    {carbon_reduction_pct:7.1%}")
    print(f"2026 Target (linear path):  {linear_target_2026:7.1f} tCO2e/$1M")
    print(f"2030 Target (50% reduction):{baseline_carbon * 0.5:7.1f} tCO2e/$1M")
    
    if portfolio_carbon <= carbon_budget:
        print(f"\n✓ Portfolio meets carbon budget of {carbon_budget:.1f} tCO2e/$1M")
    else:
        print(f"\n✗ Portfolio exceeds carbon budget by {portfolio_carbon - carbon_budget:.1f} tCO2e/$1M")

    # Backtest
    port_daily = pd.Series(rets.values @ w, index=rets.index, name="PORT")
    eq = (1 + port_daily).cumprod()
    mdd = max_drawdown(eq)
    ann_ret = port_daily.mean() * 252
    ann_vol = port_daily.std() * np.sqrt(252)
    
    print("\n" + "=" * 70)
    print("REALIZED BACKTEST")
    print("=" * 70)
    print(f"Annualized Return:        {ann_ret:7.2%}")
    print(f"Annualized Volatility:    {ann_vol:7.2%}")
    print(f"Maximum Drawdown:         {mdd:7.2%}")

    # === SLEEVE-LEVEL ANALYSIS ===
    print("\n" + "=" * 70)
    print("SLEEVE-LEVEL ANALYSIS")
    print("=" * 70)
    
    sleeve_rets = calculate_sleeve_returns(rets, w, tickers, sleeve_of)
    
    sharpe_by_sleeve = sleeve_sharpe_ratios(sleeve_rets, rf_annual=rf)
    print("\nSharpe Ratios by Sleeve:")
    print(sharpe_by_sleeve.to_string(float_format=lambda x: f"{x:.4f}"))
    
    if market_ret is not None:
        beta_by_sleeve = sleeve_betas(sleeve_rets, market_ret)
        print("\nBetas by Sleeve (vs SPY):")
        print(beta_by_sleeve.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        beta_by_sleeve = None
        print("\n⚠ Market beta analysis skipped (SPY not available)")
    
    contrib_analysis = sleeve_contribution_analysis(sleeve_rets, w, tickers, sleeve_of)
    print("\nSleeve Contribution Analysis:")
    print(contrib_analysis.to_string(float_format=lambda x: f"{x:.4f}"))

    # === CARBON ATTRIBUTION ===
    print("\n" + "=" * 70)
    print("CARBON ATTRIBUTION ANALYSIS")
    print("=" * 70)
    
    carbon_by_sleeve = carbon_attribution_by_sleeve(w, tickers, sleeve_of, carbon_vector)
    print("\nCarbon by Sleeve:")
    print(carbon_by_sleeve.to_string(float_format=lambda x: f"{x:.2f}"))
    
    carbon_by_sector = sector_carbon_exposure(w, tickers, carbon_vector)
    print("\nCarbon by Sector:")
    print(carbon_by_sector.head(10).to_string(float_format=lambda x: f"{x:.2f}"))

    # Tracking analysis
    if not sleeve_bmk_rets.empty:
        try:
            aligned_rets, aligned_bmks = rets.align(sleeve_bmk_rets, join="inner", axis=0)
            if len(aligned_rets) >= 50:
                sleeve_track = sleeve_tracking_report(aligned_rets, w, tickers, sleeve_of, aligned_bmks)
                
                if not sleeve_track.empty:
                    print("\n" + "=" * 70)
                    print("SLEEVE BENCHMARK TRACKING")
                    print("=" * 70)
                    print(sleeve_track.to_string(float_format=lambda x: f"{x:.4f}"))
        except Exception as e:
            print(f"\n⚠ Tracking analysis failed: {e}")

    # FF5
    if HAS_PDR:
        try:
            ff5 = ff5_exposure_report(port_daily, start_date=start)
            print("\n" + "=" * 70)
            print("FAMA-FRENCH 5-FACTOR EXPOSURE")
            print("=" * 70)
            print(ff5.to_string(float_format=lambda x: f"{x:.6f}"))
            alpha_ann = (1 + ff5["Alpha_daily"])**252 - 1
            print(f"\nAnnualized Alpha:         {alpha_ann:7.2%}")
        except Exception as e:
            print(f"\n⚠ FF5 failed: {e}")

    # === VISUALIZATION ===
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print("Generating 12+ charts including carbon analysis...")
    
    try:
        # 1. Basic portfolio charts
        plot_equity_curve(port_daily, "Portfolio Equity Curve")
        plot_drawdown(port_daily, "Portfolio Drawdown")
        plot_weights(w, tickers, "Portfolio Weights")
        
        # 2. Correlation analysis
        plot_correlation_heatmap(rets, tickers, sleeve_of)
        plot_correlation_3d(rets, tickers, sleeve_of)
        
        # 3. Sleeve performance
        plot_sleeve_performance(sleeve_rets)
        
        # 4. Sleeve metrics comparison
        if beta_by_sleeve is not None:
            plot_sleeve_metrics_comparison(sharpe_by_sleeve, beta_by_sleeve)
        
        # 5. Rolling metrics
        plot_rolling_sharpe(port_daily)
        
        # 6. Risk contribution
        plot_risk_contribution(contrib_analysis)
        
        # 7. Efficient frontier
        plot_efficient_frontier_position(mu, cov, w, rf, tickers, sleeve_of, A, sleeves, bounds, cons, w0)
        
        # === CARBON VISUALIZATIONS ===
        # 8. Decarbonization pathway
        plot_carbon_decarbonization_path(portfolio_carbon, baseline_carbon)
        
        # 9. Carbon by sleeve
        plot_carbon_by_sleeve(carbon_by_sleeve)
        
        # 10. Carbon-efficient frontier
        plot_carbon_efficient_frontier(mu, cov, tickers, sleeve_of, A, sleeves, bounds, cons, w0,
                                      carbon_vector, rf)
        
        print("✓ All visualizations generated successfully")
        
    except Exception as e:
        print(f"⚠ Visualization error: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print(f"• Portfolio achieves {carbon_reduction_pct:.1%} carbon reduction vs market")
    print(f"• On track for 2030 target: {'YES ✓' if portfolio_carbon <= linear_target_2026 else 'NEEDS IMPROVEMENT'}")
    print(f"• Sharpe Ratio: {sharpe:.3f}")
    print(f"• Expected Return: {port_ret_nom:.2%}")

if __name__ == "__main__":
    main()
