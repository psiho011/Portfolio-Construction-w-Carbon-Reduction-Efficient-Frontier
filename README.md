# Pension Portfolio Builder - Carbon Reduction Optimizer

A Python-based portfolio optimization tool designed for pension funds committed to achieving **50% carbon reduction by 2030** while maximizing risk-adjusted returns.

## 🎯 What This Does

This script helps you build an optimal investment portfolio that:
- ✅ Maximizes Sharpe ratio (risk-adjusted returns)
- ✅ Meets sleeve allocation constraints (e.g., 15-25% US Equity)
- ✅ Stays within your carbon budget (on path to 50% reduction by 2030)
- ✅ Provides comprehensive carbon attribution and tracking

## 📊 Key Features

### Portfolio Optimization
- Multi-sleeve portfolio construction with customizable bounds
- Mean-variance optimization using historical data + capital market assumptions
- Fama-French 5-factor analysis
- Rolling performance metrics
- Efficient frontier visualization

### Carbon Tracking & Reduction
- **Carbon footprint calculation** (tCO2e per $1M invested)
- **Linear decarbonization pathway** from 2024 baseline to 2030 target
- **Carbon budget constraints** in the optimizer
- **Attribution analysis** by sleeve and sector
- **3 visualization charts** showing carbon performance

### Sleeve Categories
1. **US_EQUITY**: Large-cap US stocks (15-25%)
2. **IG_CREDIT**: Investment-grade bonds (25-40%)
3. **TRANSITION_EQUITY**: Clean energy & climate transition (15-30%)
4. **PRIVATE_CLIMATE**: Climate solutions & infrastructure (10-20%)
5. **EM_HC_DEBT**: Emerging market high-conviction debt (5-15%)
6. **CASH**: Cash equivalents (3-7%)

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas yfinance matplotlib seaborn scipy
pip install pandas-datareader  # Optional, for Fama-French analysis
```

**Python Version**: 3.8+

### Installation

1. Download `Wharton_Portfolio_Carbon_Optimizer_v06.py`
2. Place it in your working directory
3. Run it:

```bash
python Wharton_Portfolio_Carbon_Optimizer_v06.py
```

### Basic Usage

The script will prompt you for inputs:

```
Start date for historical data (YYYY-MM-DD, default=2020-01-01): 
Max weight per ticker (default=0.05): 
Risk-free rate (default=0.04): 
Expected inflation (default=0.025): 

Max carbon intensity (default=176.0): 
Apply carbon constraint in optimization? (y/n, default=y): y

Enter tickers for each sleeve (comma-separated)
US_EQUITY stocks: AAPL, MSFT, GOOGL
TRANSITION_EQUITY stocks: TSLA, ENPH, ICLN
IG_CREDIT ETF(s): LQD, AGG
EM_HC_DEBT ETF(s): EMB
CASH ETF(s): BIL
PRIVATE_CLIMATE proxy ticker (optional): PSP
```

**Tip**: Press Enter to accept defaults for most inputs.

## 📈 Output & Reports

### Console Output

The script provides detailed text reports on:

1. **Portfolio Weights** - Top 30 holdings
2. **Expected Statistics** - Return, volatility, Sharpe ratio
3. **Carbon Footprint** - Current vs target carbon intensity
4. **Sleeve Allocation** - Validation against bounds
5. **Sleeve Performance** - Sharpe ratios, betas, contributions
6. **Carbon Attribution** - By sleeve and by sector
7. **Fama-French Factors** - 5-factor model exposures with interpretations
8. **Backtest Results** - Realized return, volatility, max drawdown

### Visualizations (13 Charts)

**Portfolio Performance:**
1. Equity curve (cumulative returns)
2. Drawdown chart
3. Portfolio weights (top holdings)
4. Correlation heatmap
5. 3D correlation surface

**Sleeve Analysis:**
6. Sleeve performance over time
7. Sleeve metrics comparison (Sharpe & Beta)
8. Rolling Sharpe ratio
9. Risk contribution by sleeve
10. Efficient frontier with optimal portfolio

**Carbon Analysis:**
11. **Decarbonization pathway** - Shows progress toward 2030 target
12. **Carbon by sleeve** - Attribution and pie chart
13. **Carbon-efficient frontier** - Risk/return colored by carbon

**Factor Analysis:**
14. **Fama-French 5-Factor exposures** - Bar chart with interpretations

## 🌍 Carbon Methodology

### Carbon Intensity Estimation

The script estimates carbon intensity (tCO2e per $1M invested) using:

1. **Sleeve-based estimates** for low-carbon allocations:
   - TRANSITION_EQUITY: 30% of market average
   - PRIVATE_CLIMATE: 20% of market average
   - IG_CREDIT: 60% of market average

2. **Sector-based estimates** for traditional equities:
   - Energy: 850 tCO2e/$1M
   - Utilities: 750 tCO2e/$1M
   - Technology: 95 tCO2e/$1M
   - Financials: 80 tCO2e/$1M
   - etc.

3. **Market baseline**: 220 tCO2e/$1M (illustrative average)

### Decarbonization Pathway

**Goal**: 50% reduction by 2030 (from 2024 baseline)

**Linear pathway**:
- 2024: 220 tCO2e/$1M (baseline)
- 2025: 202 tCO2e/$1M (8% reduction)
- **2026**: **176 tCO2e/$1M (20% reduction)** ← Current target
- 2027: 165 tCO2e/$1M (25% reduction)
- 2028: 147 tCO2e/$1M (33% reduction)
- 2029: 129 tCO2e/$1M (41% reduction)
- **2030**: **110 tCO2e/$1M (50% reduction)** ← Final target

### Carbon Constraint in Optimization

When enabled (`y` at prompt), the optimizer adds:

```
Portfolio Carbon Intensity ≤ Carbon Budget
```

This ensures the portfolio meets your decarbonization target while still maximizing risk-adjusted returns.

## 📊 Understanding the Outputs

### Key Metrics Explained

**Portfolio Statistics:**
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
- **Expected Return**: Annualized nominal return
- **Volatility**: Standard deviation of returns (lower is less risky)

**Carbon Metrics:**
- **Portfolio Carbon Intensity**: Weighted average carbon of holdings
- **Reduction from Baseline**: % reduction vs market average (220 tCO2e/$1M)
- **On Track Status**: Whether you're meeting the linear pathway to 2030

**Fama-French Factors:**
- **Mkt-RF**: Market beta (0.85 = less volatile than market, 1.15 = more volatile)
- **SMB**: Size tilt (negative = large cap, positive = small cap)
- **HML**: Value/growth (positive = value, negative = growth)
- **RMW**: Quality tilt (positive = profitable firms, negative = unprofitable)
- **CMA**: Investment style (positive = conservative, negative = aggressive)
- **Alpha**: Excess return not explained by factors

### Example Interpretation

```
Portfolio Carbon Intensity:   145.3 tCO2e/$1M
Reduction from Baseline:       34.0%
2026 Target (linear path):    176.0 tCO2e/$1M
On track for 2030 target: YES ✓
```

**Translation**: "We've achieved 34% carbon reduction (beating our 20% target for 2026), putting us ahead of schedule for the 50% reduction by 2030."

```
Mkt-RF  = 0.850  → Market beta (defensive)
SMB     = -0.120 → Size tilt (large cap)
HML     = 0.080  → Value vs Growth (value)
RMW     = 0.150  → Profitability (quality)
```

**Translation**: "Our portfolio is less volatile than the market (0.85 beta), tilted toward large-cap value stocks with high profitability. This aligns with our conservative pension mandate."

## ⚙️ Configuration & Customization

### Sleeve Bounds

Edit in the script to change allocation ranges:

```python
SLEEVE_BOUNDS = {
    "US_EQUITY": (0.15, 0.25),      # 15-25% US stocks
    "IG_CREDIT": (0.25, 0.40),      # 25-40% investment-grade bonds
    "TRANSITION_EQUITY": (0.15, 0.30),  # 15-30% clean energy
    "PRIVATE_CLIMATE": (0.10, 0.20),    # 10-20% climate solutions
    "EM_HC_DEBT": (0.05, 0.15),     # 5-15% emerging markets
    "CASH": (0.03, 0.07),           # 3-7% cash
}
```

### Capital Market Assumptions

Expected returns by sleeve (used in optimization):

```python
SLEEVE_CMA_NOMINAL = {
    "US_EQUITY": 0.065,         # 6.5% expected return
    "IG_CREDIT": 0.035,         # 3.5%
    "TRANSITION_EQUITY": 0.065, # 6.5%
    "PRIVATE_CLIMATE": 0.070,   # 7.0%
    "EM_HC_DEBT": 0.060,        # 6.0%
    "CASH": 0.030,              # 3.0%
}
```

### Carbon Targets

Adjust your reduction targets:

```python
CARBON_BASELINE_YEAR = 2024
CARBON_TARGET_YEAR = 2030
CARBON_REDUCTION_TARGET = 0.50  # 50% reduction (0.60 = 60%, etc.)
MARKET_AVERAGE_CARBON_INTENSITY = 220.0  # Baseline carbon intensity
```

### Sector Carbon Intensities

For more accurate carbon estimates, update sector intensities with actual data:

```python
SECTOR_CARBON_INTENSITY = {
    "Energy": 850.0,              # tCO2e per $1M revenue
    "Utilities": 750.0,
    "Information Technology": 95.0,
    # ... add more sectors
}
```

## 🔧 Troubleshooting

### Common Issues

**"No data returned from Yahoo Finance"**
- Check ticker symbols are valid
- Ensure you have internet connection
- Try a later start date (e.g., 2021-01-01)

**"Not enough usable tickers"**
- Need at least 5 tickers with sufficient historical data
- Increase the date range (earlier start date)
- Check that tickers have trading history for your date range

**"FF5 failed"**
- Install: `pip install pandas-datareader`
- This is optional; portfolio optimization still works without it

**Optimization warnings**
- May occur with very tight constraints
- Try loosening sleeve bounds slightly
- Increase max weight per ticker (e.g., 0.10 instead of 0.05)

**Carbon constraint cannot be met**
- Your carbon budget may be too aggressive for the given sleeve bounds
- Either: (1) Increase carbon budget, or (2) Adjust sleeve bounds to allow more low-carbon sleeves

## 📚 Technical Details

### Optimization Approach

1. **Objective**: Maximize Sharpe ratio (risk-adjusted return)
2. **Method**: Sequential Least Squares Programming (SLSQP)
3. **Constraints**:
   - Sum of weights = 100%
   - Sleeve bounds (e.g., 15-25% US equity)
   - Individual ticker bounds (e.g., max 5% per holding)
   - Carbon budget (optional)

### Risk Model

- **Historical returns**: Used to estimate covariance matrix
- **Expected returns**: 75% capital market assumptions + 25% historical
- **Annualization**: 252 trading days per year

### Data Sources

- **Price data**: Yahoo Finance (via `yfinance`)
- **Fama-French factors**: Ken French Data Library (via `pandas-datareader`)
- **Carbon estimates**: Sector-based approximations (update with real data)

## 🎯 Use Cases

### Investment Committee Presentation

Use the visualizations to show:
1. **Decarbonization pathway chart** → "We're ahead of our 2026 target"
2. **Carbon by sleeve pie chart** → "Climate sleeves contribute only 15% of carbon despite 35% weight"
3. **Efficient frontier** → "Optimal portfolio balances return, risk, and carbon"
4. **FF5 factor chart** → "Our quality tilt aligns with long-term pension needs"

### Quarterly Reporting

Track progress with:
- Carbon intensity trend over time
- Sleeve performance vs benchmarks
- Factor exposures (style drift monitoring)
- Realized vs expected returns

### Scenario Analysis

Run multiple times with different inputs:
- **Aggressive**: Carbon budget = 110 (jump to 2030 target now)
- **Baseline**: Carbon budget = 176 (stay on track)
- **Conservative**: Carbon budget = 200 (slower transition)

Compare Sharpe ratios and see the "cost" of faster decarbonization.

## 🔮 Future Enhancements

Potential additions (not yet implemented):
- [ ] Real-time ESG data API integration (MSCI, Sustainalytics)
- [ ] Scope 1/2/3 emissions breakdown
- [ ] Green bond identification within IG_CREDIT
- [ ] Carbon offset optimization
- [ ] Climate scenario analysis (1.5°C, 2°C pathways)
- [ ] Quarterly rebalancing simulator
- [ ] PDF report generation
- [ ] Excel output for further analysis

## 📞 Support & Contact

For questions about:
- **Portfolio methodology**: Contact your investment team
- **Carbon calculations**: Review TCFD/PCAF guidelines
- **Technical issues**: Check Python dependencies and data sources

## 📄 License & Disclaimer

This tool is for **educational and analytical purposes only**.

- Past performance does not guarantee future results
- Carbon estimates are approximations; use actual data when available
- Consult qualified professionals before making investment decisions
- Fama-French analysis requires `pandas-datareader` package

---

**Version**: 0.6  
**Last Updated**: January 31, 2026  
**Authors**: mpsih, Claude (Anthropic)

---

## 🚦 Quick Reference

### Minimum Example

```python
# Minimal inputs for testing:
# Start date: 2020-01-01
# Max weight: 0.05
# Risk-free rate: 0.04
# Inflation: 0.025
# Carbon budget: 176 (default)
# Apply carbon constraint: y

# Tickers:
# US_EQUITY: AAPL, MSFT, JPM
# TRANSITION_EQUITY: TSLA, ICLN
# IG_CREDIT: LQD
# EM_HC_DEBT: EMB
# CASH: BIL
# PRIVATE_CLIMATE: (leave blank or enter PSP)
```

### Key Files Created

After running, you'll have:
- 13+ matplotlib charts (displayed during run)
- Console output with all statistics
- No files saved by default (charts shown interactively)

**To save charts**: Modify the `plt.show()` calls to `plt.savefig('filename.png')`

---

**Happy Optimizing! 🌱💰**
