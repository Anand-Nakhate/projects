# Liquid Alternatives Fund-of-Funds Regime Engine
## Ultimate State-of-the-Art Design Document

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Architecture](#2-data-architecture)
3. [Feature Engineering — The Signal Stack](#3-feature-engineering)
4. [Model Architecture — The Regime Engine](#4-model-architecture)
5. [Strategy-Regime Mapping](#5-strategy-regime-mapping)
6. [Fund-Level Regime Analysis](#6-fund-level-regime-analysis)
7. [Allocation Engine](#7-allocation-engine)
8. [Monitoring & Alert System](#8-monitoring--alert-system)
9. [Backtesting Framework](#9-backtesting-framework)
10. [Build Phases](#10-build-phases)

---

## 1. System Overview

### Philosophy

The system is built on three core principles:

1. **No single model is sufficient.** Regime detection is an ill-defined problem — different
   models capture different dynamics. The ensemble of a Statistical Jump Model, Bayesian HMM,
   and BOCPD changepoint detector provides robustness that no individual model achieves.

2. **Regimes are not just about returns.** The best regime models use the full market
   microstructure: dimensionality collapse, correlation surprise, volatility term structure,
   credit conditions, and yield curve shape. Returns are a lagging symptom — these signals
   are the leading cause.

3. **The allocation decision matters more than the regime label.** The system doesn't just
   say "we're in Crisis" — it outputs a probability distribution over regimes, strategy-
   conditional expected returns and risks, and a regime-aware allocation with quantified
   confidence.

### Target Regime Taxonomy (4 States)

| Regime | Duration | Market Signature | Macro Backdrop |
|--------|----------|-----------------|----------------|
| **Growth / Risk-On** | 6-18 months | Rising equity, tight spreads, low/falling vol, steep curve | Expanding PMI, positive earnings, accommodative-to-neutral policy |
| **Slowdown / Late Cycle** | 3-9 months | Flat/choppy equity, widening spreads, rising vol, flattening curve | Decelerating PMI, tightening policy, peak margins |
| **Crisis / Dislocation** | 1-6 months | Equity drawdown, spread blowout, vol explosion, correlation spike | Negative PMI, liquidity withdrawal, deleveraging |
| **Recovery / Reflation** | 3-9 months | Equity rebounds, spreads tightening from wides, vol falling from highs | PMI trough, policy easing, positioning washout |

These 4 states are validated by the model (not imposed), but we initialize the Jump Model
and HMM with these priors. The data ultimately determines the regime boundaries.

### Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         DATA LAYER (Daily + Monthly)                    ║
║  Bloomberg API  │  Fund Returns  │  HF Indices  │  Macro Series        ║
╚════════════════════════════╦═════════════════════════════════════════════╝
                             ║
                             ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                    FEATURE ENGINEERING (26 Signals)                      ║
║                                                                          ║
║  ┌──────────────────┐ ┌───────────────┐ ┌─────────────────────────────┐ ║
║  │  DIMENSIONALITY  │ │  TURBULENCE   │ │     VOLATILITY REGIME       │ ║
║  │  • Absorption    │ │  • Mahalanobis│ │  • VIX level + %ile         │ ║
║  │    Ratio + ΔAR   │ │  • Correlation│ │  • VIX term structure       │ ║
║  │  • ENB (Meucci)  │ │    Surprise   │ │  • Realized vs Implied     │ ║
║  │  • Eff. Dim.     │ │  • Systemic   │ │  • VVIX                    │ ║
║  │  • Eigenvalue    │ │    Risk Score │ │  • Cross-asset vol corr    │ ║
║  │    Dispersion    │ │               │ │                             │ ║
║  └──────────────────┘ └───────────────┘ └─────────────────────────────┘ ║
║  ┌──────────────────┐ ┌───────────────┐ ┌─────────────────────────────┐ ║
║  │  CREDIT/LIQUIDITY│ │  YIELD CURVE  │ │     CROSS-ASSET / BEHAV.   │ ║
║  │  • IG/HY OAS     │ │  • Level/Slope│ │  • Stock-Bond correlation  │ ║
║  │  • HY-IG diff.   │ │    /Curvature │ │  • Cross-asset momentum    │ ║
║  │  • SOFR-FF spread│ │  • Real rates │ │  • Dispersion index        │ ║
║  │  • CP spreads    │ │  • Fin. cond. │ │  • Sentiment / positioning │ ║
║  │  • X-ccy basis   │ │  • PMI regime │ │  • Fund flows              │ ║
║  └──────────────────┘ └───────────────┘ └─────────────────────────────┘ ║
║                                                                          ║
║  All signals → percentile rank (expanding window) + z-score              ║
╚════════════════════════════╦═════════════════════════════════════════════╝
                             ║
                             ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                       MODEL LAYER (Ensemble)                             ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  TIER 1: Statistical Jump Model (PRIMARY)                        │   ║
║  │  • K=4 states, jump penalty λ (cross-validated)                  │   ║
║  │  • Sparse variant for automatic feature selection                │   ║
║  │  • Expanding window, monthly retrain                             │   ║
║  │  • Output: state assignment + distance-to-centroid soft probs    │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  TIER 2: Bayesian HMM with Student-t Emissions (SECONDARY)      │   ║
║  │  • K=4 states, fat-tailed emission distributions                 │   ║
║  │  • Variational Bayes estimation                                  │   ║
║  │  • Output: P(state|data) + transition matrix                     │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  TIER 3: BOCPD — Bayesian Online Changepoint Detection (ALERT)  │   ║
║  │  • Runs on DAILY data (5 key signals)                            │   ║
║  │  • Online/streaming, no retrain needed                           │   ║
║  │  • Output: P(changepoint) at each timestep                       │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  ENSEMBLE: Confidence-Weighted Consensus                         │   ║
║  │  • Agree → high confidence regime assignment                     │   ║
║  │  • Disagree → "transition/uncertain" flag                        │   ║
║  │  • BOCPD override → forced transition alert                      │   ║
║  │  • Output: P(Growth), P(Slowdown), P(Crisis), P(Recovery)       │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚════════════════════════════╦═════════════════════════════════════════════╝
                             ║
                             ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                      APPLICATION LAYER                                   ║
║                                                                          ║
║  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  ║
║  │ STRATEGY-REGIME    │  │  FUND-LEVEL        │  │  ALLOCATION      │  ║
║  │ MAPPING            │  │  ANALYSIS          │  │  ENGINE          │  ║
║  │                    │  │                    │  │                  │  ║
║  │ • Conditional      │  │ • Regime beta      │  │ • BL w/ regime  │  ║
║  │   returns/risk     │  │ • Conditional α    │  │   views         │  ║
║  │ • Conditional      │  │ • Style drift      │  │ • Regime-switch │  ║
║  │   Sharpe ratios    │  │   detection        │  │   risk parity   │  ║
║  │ • Transition       │  │ • Transition       │  │ • Confidence    │  ║
║  │   performance      │  │   sensitivity      │  │   scaling       │  ║
║  │ • Conditional      │  │ • Tail risk by     │  │ • Liquidity &   │  ║
║  │   correlations     │  │   regime           │  │   turnover      │  ║
║  │ • Fung-Hsieh α     │  │ • Conditional IR   │  │   constraints   │  ║
║  │   by regime        │  │                    │  │                  │  ║
║  └────────────────────┘  └────────────────────┘  └──────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Data Architecture

### 2.1 Market Data (Daily — Bloomberg)

| Category | Tickers / Series | Purpose |
|----------|-----------------|---------|
| **Equity Indices** | SPX, MXWO, MXEF, RTY, SX5E, NKY, sector ETFs (XLF, XLK, XLE, etc.) | Equity regime, sector rotation |
| **Govt Bonds** | US 2Y, 5Y, 10Y, 30Y yields; DE 10Y; JP 10Y; TIPS 5Y/10Y | Yield curve decomposition, real rates |
| **Credit** | CDX IG, CDX HY, iTraxx Main, IG OAS (LUACOAS), HY OAS (LF98OAS) | Credit regime, stress detection |
| **Volatility** | VIX, VIX3M, VVIX, MOVE, JPMVXYG7 (FX vol), SKEW | Vol regime, term structure, skew |
| **Commodities** | CL1 (WTI), GC1 (Gold), HG1 (Copper), BCOM | Inflation proxy, growth proxy (copper/gold) |
| **FX** | DXY, EURUSD, USDJPY, JPYUSD carry, EMFX basket (CEW) | Dollar regime, risk appetite |
| **Liquidity** | SOFR-FF spread, 3M CP-OIS, TED spread, x-ccy basis (EUR, JPY) | Funding stress, liquidity regime |
| **Flows/Position** | CFTC COT net positioning (SPX, 10Y, Gold, EUR), EPFR fund flows | Positioning extremes, sentiment |

**Total: ~50-60 daily series**

### 2.2 Macro Data (Monthly — Bloomberg / FRED)

| Series | Frequency | Role |
|--------|-----------|------|
| ISM Manufacturing PMI + Services PMI | Monthly | Growth regime (above/below 50 + direction) |
| US CPI YoY + Core CPI | Monthly | Inflation regime |
| Michigan Inflation Expectations (1Y, 5Y) | Monthly | Inflation expectations |
| US NFP + Unemployment Rate | Monthly | Labor market regime |
| US Retail Sales MoM | Monthly | Consumer health |
| Conference Board Leading Economic Index | Monthly | Recession probability |
| Chicago Fed NFCI | Weekly | Financial conditions composite |
| OECD Composite Leading Indicator | Monthly | Global growth regime |
| Fed Funds Rate + ECB Depo Rate | Meeting dates | Policy regime |
| Fed Balance Sheet (WALCL) | Weekly | Liquidity regime |

**Total: ~15-20 macro series**

### 2.3 Hedge Fund Index Data (Monthly)

| Index Family | Sub-Indices | Purpose |
|-------------|------------|---------|
| **HFRI** | Equity Hedge, Event Driven, Macro, Relative Value, + sub-strategies | Strategy regime mapping |
| **Credit Suisse / Tremont** | L/S Equity, Event Driven, Global Macro, Managed Futures, Multi-Strat | Cross-reference |
| **SG** | SG CTA Index, SG Trend Index, SG Short-Term Traders | CTA/trend regime |
| **CBOE** | Eurekahedge Long Vol, Short Vol indices | Vol strategy regime |

**Total: ~25-30 monthly index series**

### 2.4 Fund Data

| Data Point | Frequency | Source |
|-----------|-----------|--------|
| NAV / Return | Daily, Weekly, or Monthly (mixed) | Admin / internal systems |
| Strategy Classification | Static (with updates) | Internal tagging |
| AUM | Monthly | Fund reporting |
| Redemption Terms | Static | DDQ / side letters |
| Factor Exposures | Monthly (if available) | Risk system / return-based |
| Holdings (if available) | Monthly/Quarterly | Transparency reports |

### 2.5 Factor Data (Monthly)

| Factor Set | Components | Purpose |
|-----------|-----------|---------|
| **Fung-Hsieh 7 Factors** | Equity mkt, size spread, Δ10Y yield, Δcredit spread, bond/FX/commodity trend (lookback straddles) | Standard HF factor model |
| **Fama-French 5 + Momentum** | MKT, SMB, HML, RMW, CMA, MOM | Equity factor exposure |
| **AQR Factors** | Value, Momentum, Carry, Defensive — across equities, FI, FX, commodities | Multi-asset factor exposure |
| **Volatility Risk Premium** | VIX - 30-day realized vol (SPX) | Vol selling regime |
| **Merger Arb Spread** | Deal spread index | Event driven regime |
| **Convertible Arb Spread** | Cheapness of converts vs theoretical value | RV regime |

### 2.6 Data Processing Rules

1. **Frequency alignment**: All daily data aggregated to monthly for regime model training
   (end-of-month value, monthly return, monthly average, monthly change — depends on series type)
2. **Mixed-frequency fund data**: Use last available observation within the month. If weekly,
   use last week's return compounded. Flag funds with >10 day reporting lag.
3. **Expanding window standardization**: All features converted to percentile rank on
   expanding window (minimum 36 months) to avoid look-ahead bias
4. **Missing data**: Forward-fill up to 3 months. Beyond that, exclude or interpolate.
   Never backfill.
5. **Outlier treatment**: Winsorize at 1st/99th percentile before model input (not before
   turbulence calculation — turbulence NEEDS outliers)

---

## 3. Feature Engineering — The Signal Stack

### 3.1 Dimensionality / Market Structure Signals

#### Signal 1: Absorption Ratio (AR) — Kritzman, Li, Page, Rigobon (2010)

**What it measures**: Fraction of total variance explained by top eigenvectors. Captures how
"unified" or "concentrated" risk sources are.

**Calculation**:
```
1. Take trailing 500 daily returns for N cross-asset series
   (use: SPX, RTY, SX5E, MXEF, 10Y UST total return, IG credit, HY credit,
    Gold, Oil, DXY — approximately 10-15 series)
2. Compute correlation matrix Σ
3. Eigendecompose: Σ = QΛQ'
4. Sort eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ
5. AR = Σᵢ₌₁ⁿ λᵢ / Σⱼ₌₁ᴺ λⱼ   where n = ⌊N/5⌋
6. Compute ΔAR = (AR_t - AR_{t-15}) / σ(ΔAR)  [15-day change, standardized]
```

**Regime signal**:
- AR level > 80th percentile → systemic risk elevated → Slowdown/Crisis regime
- ΔAR > +1.0σ → rapid risk concentration → imminent regime transition
- ΔAR < -1.0σ → risk dispersing → transitioning toward Growth/Recovery

**Parameters**: 500-day rolling window, N/5 eigenvectors, 15-day change horizon

---

#### Signal 2: Effective Number of Bets (ENB) — Meucci (2009)

**What it measures**: Shannon entropy of the eigenvalue distribution. Captures the "effective
dimensionality" of risk — how many independent risk sources are active.

**Calculation**:
```
1. Same correlation matrix as AR (trailing 252 daily returns)
2. Compute normalized eigenvalues: pᵢ = λᵢ / Σλ
3. ENB = exp(-Σ pᵢ · ln(pᵢ))    [exponential of Shannon entropy]
```

**Regime signal**:
- ENB ranges from 1 (single dominant factor) to N (fully diversified)
- ENB < 30th percentile → concentrated risk → Crisis precursor
- ENB > 70th percentile → diversified → Growth/Recovery
- Rapid ENB decline (>1σ drop in 1 month) → regime transition warning

**Parameters**: 252-day rolling window

---

#### Signal 3: Effective Dimensionality — Participation Ratio

**What it measures**: Alternative to ENB, more robust to extreme eigenvalues.

**Calculation**:
```
d_eff = (Σλᵢ)² / Σ(λᵢ²)
```

**Regime signal**: Same interpretation as ENB but smoother. Use as confirmation signal.

---

#### Signal 4: Eigenvalue Dispersion / Dominance

**What it measures**: How dominant the first principal component is.

**Calculation**:
```
1. λ₁/λ₂ ratio (first-to-second eigenvalue)
2. Number of eigenvalues exceeding Marchenko-Pastur upper bound:
   λ_MP_max = σ²(1 + √(N/T))²
   where σ² = average eigenvalue, N = assets, T = observations
3. Count: K_significant = #{λᵢ > λ_MP_max}
```

**Regime signal**:
- λ₁/λ₂ > 3 → single-factor market → Crisis/macro-driven
- K_significant declining → market becoming more factor-concentrated
- Cross-reference: when K_significant drops AND AR rises → high systemic risk

---

### 3.2 Turbulence / Dislocation Signals

#### Signal 5: Financial Turbulence — Kritzman & Li (2010)

**What it measures**: Mahalanobis distance of current returns from historical distribution.
Spikes when asset prices move in "unusual" ways — including extreme moves, decorrelation of
normally correlated assets, and convergence of normally uncorrelated assets.

**Calculation**:
```
1. Rolling mean μ and covariance Σ from trailing 252 daily returns
   (same cross-asset universe as AR)
2. d_t = (r_t - μ)' · Σ⁻¹ · (r_t - μ)
3. Under null: d_t ~ χ²(N)  →  compute p-value
4. Smooth with 21-day EMA for monthly signal
```

**Regime signal**:
- d_t > 95th percentile → turbulent → Crisis
- d_t > 75th percentile → elevated → Slowdown
- Sustained d_t < 50th percentile → calm → Growth
- Key property: this fires for BOTH high-vol events AND unusual correlation events

---

#### Signal 6: Correlation Surprise — Kinlaw & Turkington (2012)

**What it measures**: The correlation component of turbulence, orthogonal to volatility.
Isolates whether correlations are behaving unusually independent of vol.

**Calculation**:
```
1. Decompose turbulence into:
   - Volatility component: replace Σ with diagonal-only covariance
     d_vol = (r_t - μ)' · D⁻¹ · (r_t - μ)   where D = diag(Σ)
   - Correlation surprise: residual from regression of d_t on d_vol
     CS_t = d_t - β · d_vol  (or orthogonalized component)
2. Alternative (cleaner):
   - Standardize returns: z_t = D^(-1/2) · (r_t - μ)
   - Correlation surprise: CS_t = z_t' · R⁻¹ · z_t
     where R = correlation matrix (not covariance)
```

**Regime signal**:
- High CS → correlations are unusual → regime transition in progress
- CS spikes BEFORE vol spikes → leading indicator
- Periods of high CS lead to higher risk and lower risk premia (Kinlaw & Turkington empirical result)
- This signal is orthogonal to vol → provides unique information

---

#### Signal 7: Systemic Risk Composite — Kinlaw, Kritzman, Turkington

**What it measures**: Combined connectivity and fragility of the financial system.

**Calculation**:
```
Systemic_Risk = w₁ · AR_percentile + w₂ · Turbulence_percentile + w₃ · CS_percentile
(equal weights as default, or PCA on these three signals)
```

**Regime signal**: Composite provides a more stable signal than any single component.

---

### 3.3 Volatility Regime Signals

#### Signal 8: VIX Level (Percentile Rank)

```
VIX_pctile = percentile_rank(VIX_t, VIX_{t-504:t})  [2-year expanding window]
```
- < 25th %ile: Low vol regime (complacency / Growth)
- 25th-75th %ile: Normal
- > 75th %ile: High vol regime (Slowdown/Crisis)
- > 95th %ile: Extreme (Crisis)

#### Signal 9: VIX Term Structure

```
VTS = VIX / VIX3M  (or VX1/VX2 futures ratio)
```
- VTS < 0.9: Deep contango → complacency, carry-friendly (Growth)
- VTS 0.9-1.0: Normal
- VTS > 1.0: Backwardation → near-term fear exceeds longer-term (Crisis)
- Persistent backwardation (>5 days) = strongest Crisis signal

#### Signal 10: Realized-Implied Vol Spread

```
RIV = VIX - RealizedVol_30d(SPX)   [Yang-Zhang estimator for realized]
```
- Large positive RIV: Market pricing in future risk → precautionary (Slowdown)
- Large negative RIV: Realized catching up to implied → in crisis already
- Near zero: Equilibrium (Growth or Recovery)

#### Signal 11: VVIX (Vol of Vol)

```
VVIX_pctile = percentile_rank(VVIX_t, expanding window)
```
- VVIX > 80th %ile: Uncertainty about uncertainty → fragile regime
- Cross with VIX: High VIX + High VVIX = unstable crisis, High VIX + Low VVIX = stable fear

#### Signal 12: Cross-Asset Volatility Correlation

```
VolCorr = corr(VIX, MOVE, 60 days)   [equity vol vs bond vol correlation]
+ corr(VIX, CVIX, 60 days)           [equity vol vs FX vol]
```
- All three rising together = systemic stress → Crisis
- Divergence (VIX up, MOVE down) = equity-specific, not systemic → Slowdown

---

### 3.4 Credit / Liquidity Signals

#### Signal 13: Credit Spread Level + Momentum

```
IG_pctile = percentile_rank(IG_OAS, expanding)
HY_pctile = percentile_rank(HY_OAS, expanding)
Spread_momentum = (OAS_t - OAS_{t-63}) / σ(ΔOAS_63d)  [3-month change, standardized]
```
- HY OAS > 80th %ile → Credit stress → Crisis
- Rapid widening (momentum > +1.5σ) → regime deteriorating
- Tightening from wides → Recovery

#### Signal 14: Credit Quality Differentiation

```
HY_IG_diff = HY_OAS - IG_OAS
Quality_spread_pctile = percentile_rank(HY_IG_diff, expanding)
```
- Compression (low HY-IG diff) → market not discriminating → complacency (late Growth)
- Expansion → flight to quality → Slowdown/Crisis

#### Signal 15: Funding/Liquidity Stress

```
Composite: average of percentile ranks of:
  - SOFR-FF spread
  - 3M commercial paper - OIS spread
  - EUR/USD 3M cross-currency basis swap (inverted — more negative = more stress)
```
- Composite > 75th %ile → funding stress → Crisis precursor
- This signal leads credit spreads by 1-3 weeks

---

### 3.5 Yield Curve / Macro Signals

#### Signal 16: Yield Curve Decomposition (Nelson-Siegel or PCA)

```
Using PCA on US Treasury curve (2Y, 5Y, 10Y, 30Y):
  Level = PC1 score    (parallel shift — rate environment)
  Slope = PC2 score    (steepening/flattening — growth expectations)
  Curvature = PC3 score (butterfly — stress/dislocation)

Or Nelson-Siegel:
  Level (β₀), Slope (β₁), Curvature (β₂)

Key derived signals:
  10Y-2Y spread: > 0 = normal, < 0 = inverted = recession signal
  2s5s10s butterfly = 2*(5Y) - 2Y - 10Y: spikes during stress
```

**Regime mapping**:
- Steep curve + rising level → Growth (economy expanding, rates rising)
- Flat/inverted curve → Slowdown (recession expectations)
- Curve steepening from inversion → Recovery (policy easing)
- Curvature spike → Crisis (flight to safety distorting curve)

#### Signal 17: Real Rate Regime

```
RealRate = TIPS_5Y or TIPS_10Y yield
RealRate_direction = sign(RealRate_t - RealRate_{t-63})
```
- Negative + falling real rates → accommodative → Growth-supportive
- Positive + rising real rates → tightening → Slowdown/Crisis risk
- Most important for: duration-sensitive strategies, credit, growth equity

#### Signal 18: Financial Conditions Index

```
NFCI = Chicago Fed National Financial Conditions Index  [weekly]
NFCI_momentum = NFCI_t - NFCI_{t-13 weeks}
```
- NFCI > 0 = tighter than average → Slowdown
- NFCI < 0 = looser than average → Growth
- Rapid tightening (momentum > +1σ) → regime deterioration
- This is itself a composite of 105 measures — extremely information-rich

#### Signal 19: PMI Regime (4-Quadrant)

```
PMI_level = ISM_Manufacturing  (or Global Composite)
PMI_direction = sign(PMI_t - PMI_{t-3})

Quadrant mapping:
  PMI > 50 + Rising   → Expansion-Accelerating  → Growth
  PMI > 50 + Falling  → Expansion-Decelerating  → late Growth / Slowdown
  PMI < 50 + Falling  → Contraction-Accelerating → Crisis
  PMI < 50 + Rising   → Contraction-Decelerating → Recovery
```

---

### 3.6 Cross-Asset / Behavioral Signals

#### Signal 20: Stock-Bond Correlation

```
SB_corr = rolling_corr(SPX_daily_ret, TLT_daily_ret, 60 days)
```
- SB_corr < -0.3 → risk-driven market → bonds hedge equity → Growth/normal
- SB_corr > +0.2 → inflation/rate-driven → bonds don't hedge → regime shift
- Sign flip → major regime transition
- This is one of the most powerful regime indicators available

#### Signal 21: Cross-Asset Trend Alignment

```
For each asset (SPX, AGG, GLD, DXY, CL1):
  trend_score = sign(price / SMA_200)   [+1 or -1]

Alignment = Σ trend_scores / N_assets    [-1 to +1]
```
- Alignment > +0.6 → broad uptrend → Growth (good for trend followers)
- Alignment < -0.6 → broad downtrend → Crisis (good for trend followers too)
- Alignment near 0 → choppy/mixed → bad for trend, okay for RV

#### Signal 22: Cross-Sectional Dispersion

```
Dispersion = σ(R_sector_1, R_sector_2, ..., R_sector_11)  [cross-sectional vol of sector returns]
Dispersion_pctile = percentile_rank(Dispersion, expanding)
```
- Low dispersion + high correlation → macro-driven → bad for stock pickers
- High dispersion + low correlation → alpha opportunity → good for L/S equity
- Cross with regime: Crisis = high corr, low useful dispersion; Recovery = rising dispersion

#### Signal 23: Sentiment / Positioning Extremes

```
Composite of:
  - AAII Bull-Bear spread percentile (inverted — extreme bullishness = contrarian bearish)
  - Put/Call ratio percentile
  - CFTC net positioning in SPX futures (percentile)
  - VIX futures net positioning (percentile)
```
- Extreme bullish positioning → late Growth → vulnerable to Slowdown
- Extreme bearish positioning → capitulation → Recovery setup
- Sentiment is a contrarian timing signal, not a regime signal — use for transition timing

---

### 3.7 Feature Matrix Summary

| # | Signal | Frequency | Category | Primary Use |
|---|--------|-----------|----------|-------------|
| 1 | Absorption Ratio (AR) | Daily→Monthly | Dimensionality | Systemic risk level |
| 2 | ΔAR (standardized change) | Daily→Monthly | Dimensionality | Regime transition speed |
| 3 | ENB (Meucci) | Daily→Monthly | Dimensionality | Diversification state |
| 4 | Effective Dimensionality | Daily→Monthly | Dimensionality | Robustness check for ENB |
| 5 | λ₁/λ₂ ratio | Daily→Monthly | Dimensionality | Single-factor dominance |
| 6 | Financial Turbulence | Daily→Monthly | Turbulence | Unusual market behavior |
| 7 | Correlation Surprise | Daily→Monthly | Turbulence | Unusual correlation (ortho to vol) |
| 8 | VIX percentile | Daily→Monthly | Volatility | Vol regime level |
| 9 | VIX term structure | Daily→Monthly | Volatility | Near-term fear vs. longer-term |
| 10 | Realized-Implied spread | Daily→Monthly | Volatility | Vol expectation gap |
| 11 | VVIX percentile | Daily→Monthly | Volatility | Regime stability/fragility |
| 12 | Cross-asset vol corr | Daily→Monthly | Volatility | Systemic vs. idiosyncratic vol |
| 13 | HY OAS percentile | Daily→Monthly | Credit | Credit stress level |
| 14 | Credit spread momentum | Daily→Monthly | Credit | Rate of deterioration |
| 15 | HY-IG quality spread | Daily→Monthly | Credit | Risk discrimination |
| 16 | Funding stress composite | Daily→Monthly | Liquidity | Liquidity regime |
| 17 | Yield curve slope (10Y-2Y) | Daily→Monthly | Rates | Growth expectations |
| 18 | Yield curve curvature | Daily→Monthly | Rates | Curve stress |
| 19 | Real rate level + dir. | Daily→Monthly | Rates | Policy tightness |
| 20 | NFCI + momentum | Weekly→Monthly | Macro | Financial conditions composite |
| 21 | PMI quadrant (level × dir) | Monthly | Macro | Growth regime |
| 22 | Stock-bond correlation | Daily→Monthly | Cross-Asset | Correlation regime |
| 23 | Cross-asset trend alignment | Daily→Monthly | Cross-Asset | Trend regime |
| 24 | Cross-sectional dispersion | Daily→Monthly | Cross-Asset | Alpha opportunity regime |
| 25 | Sentiment composite | Weekly→Monthly | Behavioral | Positioning extremes |
| 26 | Systemic risk composite | Daily→Monthly | Composite | Overall risk score |

**Total: 26 features → 26-dimensional monthly feature vector for the regime model**

All features are standardized to percentile rank on an expanding window (min 36 months)
before entering the model. This ensures stationarity, comparability, and no look-ahead.

---

## 4. Model Architecture — The Regime Engine

### 4.1 Tier 1: Statistical Jump Model (PRIMARY)

**Why this is the primary model**:
- Produces more persistent state sequences than HMMs (fewer whipsaws)
- The jump penalty λ explicitly controls the persistence/responsiveness tradeoff
- Sparse variant performs automatic feature selection — critical with 26 features
- More robust to initialization than EM-based HMMs
- Demonstrated superior out-of-sample financial performance vs. HMMs
  (Nystrup et al. 2020, Shu & Mulvey 2024)

**Algorithm**:
```
Minimize: Σ_t loss(x_t, θ_{s_t}) + λ · Σ_t 𝟙(s_t ≠ s_{t-1})

Where:
  x_t     = 26-dimensional feature vector at time t
  s_t     = regime assignment ∈ {1, 2, 3, 4}
  θ_k     = centroid (mean + covariance) for regime k
  λ       = jump penalty (higher → fewer regime changes)
  loss    = squared Mahalanobis distance to cluster centroid
```

**Configuration**:
- **K = 4 states** (Growth, Slowdown, Crisis, Recovery)
  - Validate with BIC, silhouette score, and economic interpretability
  - Test K = 3 and K = 5 as robustness checks
- **Jump penalty λ**: Cross-validate on expanding window
  - Grid search: λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
  - Criterion: maximize out-of-sample regime-conditional Sharpe differential
  - Typical optimal: λ ≈ 1.0-5.0 (produces regimes lasting 3-12 months)
- **Sparse Jump Model variant**:
  - Adds L1 penalty on feature weights → automatic feature selection
  - Use this to identify which of the 26 signals are most regime-discriminating
  - Expected: ~10-15 features survive sparsification
- **Training**: Expanding window, retrain monthly
  - Minimum training window: 60 months (5 years)
  - Fit on standardized (percentile rank) features
- **Initialization**: Use k-means++ on features, then refine with jump penalty
  - Run 10 random initializations, select lowest objective

**Output**:
- Hard state assignment: s_t ∈ {1, 2, 3, 4}
- Soft probabilities: convert via distance to centroids
  ```
  P(regime_k | x_t) ∝ exp(-d(x_t, θ_k) / temperature)
  ```
  where temperature is calibrated so that confident assignments have P > 0.7
- Regime centroids: the "profile" of each regime (which signals are high/low)
- Feature importance (from sparse variant): which signals matter most

**Implementation**: Use the `jump-models` Python package (scikit-learn API)
- GitHub: https://github.com/Yizhan-Oliver-Shu/jump-models

---

### 4.2 Tier 2: Bayesian HMM with Student-t Emissions (SECONDARY)

**Why as secondary**:
- Different inductive bias → ensemble diversity
  - HMM assumes Markovian transitions (future depends only on current state)
  - Jump Model assumes persistence penalty (different structural assumption)
  - When both agree → very high confidence
- Provides a transition matrix → forward-looking regime probabilities
- Student-t emissions handle fat tails (Gaussian HMM underweights extreme observations)
- Bayesian estimation provides posterior uncertainty on parameters

**Algorithm**:
```
Standard HMM with:
  - K = 4 hidden states
  - Emission distributions: multivariate Student-t(ν, μ_k, Σ_k)
    where ν (degrees of freedom) is estimated per state
  - Transition matrix: A[i,j] = P(s_{t+1}=j | s_t=i)
  - Initial distribution: π_k = P(s_0=k)

Estimation: Variational Bayes (preferred) or MCMC
  - Variational Bayes is faster and scales better
  - MCMC (via PyMC or Stan) for full posterior if needed
```

**Configuration**:
- **K = 4** (same as Jump Model for comparability)
- **Feature set**: Same 26 features (or the sparse subset from Jump Model)
  - Optionally: use PCA to reduce to 8-10 components first
    (HMMs are less robust with high-dimensional features)
- **Student-t emissions**: Estimate ν per state
  - Expect ν ≈ 3-5 for Crisis state (very fat tails)
  - Expect ν ≈ 10-20 for Growth state (near Gaussian)
- **Prior specification** (Bayesian):
  - Dirichlet prior on transition rows: α_ii = 10, α_ij = 1 (favor self-transition)
  - Normal-Wishart prior on emission parameters
  - Weakly informative to allow data to dominate
- **Training**: Expanding window, retrain monthly (same as Jump Model)

**Output**:
- Filtered probabilities: P(s_t = k | x_{1:t}) — real-time regime assessment
- Smoothed probabilities: P(s_t = k | x_{1:T}) — retrospective (for analysis only)
- Transition matrix A: P(s_{t+1} | s_t) — critical for forward-looking allocation
- Most likely state sequence (Viterbi)
- Posterior parameter uncertainty (from Bayesian estimation)

**Forward-looking power** (unique to HMM):
```
P(regime at t+1) = Σ_k P(s_t = k | data) · A[k, :]

Example: If currently 80% Growth, 20% Slowdown, and transition matrix says
  P(Slowdown | Growth) = 0.10, P(Crisis | Slowdown) = 0.15
Then: P(Crisis at t+2) incorporates multi-step transition risk
```

**Implementation**: `hmmlearn` (Gaussian) + custom Student-t extension, or `pomegranate`

---

### 4.3 Tier 3: BOCPD — Bayesian Online Changepoint Detection (ALERT LAYER)

**Why as alert layer**:
- Runs on DAILY data (Jump Model and HMM run monthly)
- Online/streaming: no retraining needed, O(1) update per observation
- Detects regime transitions in real-time → triggers intra-month review
- Bayesian: quantifies uncertainty about whether a changepoint occurred
- Model-free: doesn't assume specific regime structure, just detects "something changed"

**Algorithm** (Adams & MacKay, 2007):
```
At each time step t, maintain posterior over run length r_t:
  P(r_t | x_{1:t})

where r_t = time since last changepoint.

Update rules:
  1. Growth probability: P(r_t = r_{t-1}+1) ∝ P(x_t | x_{after last CP}) · (1-H)
  2. Changepoint probability: P(r_t = 0) ∝ Σ_r P(x_t | CP) · H · P(r_{t-1}=r)

  where H = hazard function = P(changepoint at any given time)

Key output: P(changepoint at time t) = P(r_t = 0 | x_{1:t})
```

**Configuration**:
- **Input signals** (5 key daily series — keep it focused):
  1. VIX level
  2. HY OAS (credit spread)
  3. Absorption Ratio
  4. Stock-bond correlation (rolling 20-day)
  5. NFCI (financial conditions)
- **Hazard function**: Constant, H = 1/250 (expect ~1 changepoint per year)
  - Can be made adaptive: higher H when vol is elevated
- **Observation model**: Gaussian with unknown mean and variance
  - Use conjugate Normal-Inverse-Gamma prior → closed-form updates
  - Prior updated online with each observation (no batch retraining)
- **Run on each signal independently** → 5 changepoint probability streams
- **Composite alert**: when ≥2 signals show P(changepoint) > 0.5 simultaneously

**Alert triggers**:
- P(changepoint) > 0.7 on any single signal → yellow alert
- P(changepoint) > 0.7 on ≥2 signals → red alert → trigger regime review
- P(changepoint) > 0.9 on ≥3 signals → emergency → immediate allocation review

**Implementation**: Custom Python implementation or `bayesian-changepoint-detection` package

---

### 4.4 Ensemble: Confidence-Weighted Consensus

The three models serve different roles and combine as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ENSEMBLE LOGIC                               │
│                                                                  │
│  Jump Model regime: JM ∈ {Growth, Slowdown, Crisis, Recovery}   │
│  HMM probabilities: P_HMM(k) for k ∈ {Growth, Slowdown, ...}   │
│  BOCPD alert:       changepoint ∈ {None, Yellow, Red, Emergency} │
│                                                                  │
│  CASE 1: JM and HMM agree (same top regime)                     │
│    → Regime = agreed regime                                      │
│    → Confidence = HIGH                                           │
│    → P(regime) = 0.5·P_JM(k) + 0.5·P_HMM(k) [blended]        │
│                                                                  │
│  CASE 2: JM and HMM disagree                                    │
│    → Regime = regime with highest blended probability            │
│    → Confidence = LOW                                            │
│    → Flag: "TRANSITION/UNCERTAIN"                                │
│    → Allocation: reduce active tilts, increase diversification   │
│                                                                  │
│  CASE 3: BOCPD Red/Emergency alert (regardless of JM/HMM)       │
│    → Override confidence to LOW                                  │
│    → Flag: "CHANGEPOINT DETECTED — REGIME MAY BE SHIFTING"      │
│    → Trigger: intra-month allocation review                      │
│    → Do NOT immediately change regime label (BOCPD detects       │
│      change, but doesn't know the new regime yet)                │
│                                                                  │
│  FINAL OUTPUT at each month-end:                                 │
│    {                                                             │
│      regime: "Growth",                                           │
│      probabilities: {Growth: 0.72, Slowdown: 0.18, ...},        │
│      confidence: "HIGH",                                         │
│      transition_matrix: [[0.85, 0.10, 0.02, 0.03], ...],       │
│      bocpd_alert: "None",                                        │
│      regime_age: 7,  // months in current regime                 │
│      feature_profile: {...}  // signal values defining regime    │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Weight calibration**: Weights between JM and HMM (default 50/50) can be calibrated
on rolling out-of-sample basis. Metric: which model's regime-conditional returns have
higher discriminative power (larger Sharpe differential between regimes)?

---

## 5. Strategy-Regime Mapping

### 5.1 Strategy Universe for Liquid Alternatives

| Strategy Cluster | Sub-Strategies | Key Drivers |
|-----------------|----------------|-------------|
| **L/S Equity** | Fundamental, Quantitative, Sector, Market Neutral | Equity beta, dispersion, alpha generation |
| **Global Macro** | Discretionary, Systematic, EM Macro | Macro trends, rate differentials, policy shifts |
| **CTA / Managed Futures** | Trend-following, Short-term, Multi-strategy | Trend persistence, vol regime, cross-asset trends |
| **Event Driven** | Merger Arb, Activist, Distressed, Special Sits | Deal flow, credit cycle, corporate activity |
| **Relative Value** | Fixed Income Arb, Convertible Arb, Stat Arb, Vol Arb | Spread levels, mean reversion, vol of vol |
| **Multi-Strategy** | Multi-PM, Multi-sleeve | Diversified — performance driven by PM selection |
| **Credit** | Long/Short Credit, Structured Credit, CLOs | Credit cycle, spreads, default rates |
| **Volatility** | Long Vol, Short Vol, Dispersion, Tail Hedging | VIX level, term structure, realized-implied |

### 5.2 Conditional Performance Analysis

For each strategy cluster and each regime, compute:

```python
# Pseudocode for strategy-regime analysis
for strategy in strategies:
    for regime in [Growth, Slowdown, Crisis, Recovery]:

        # Filter returns to regime periods
        regime_returns = strategy_returns[regime_labels == regime]

        # 1. Conditional Return Distribution
        E_return[strategy][regime]  = mean(regime_returns)
        vol[strategy][regime]       = std(regime_returns)
        skew[strategy][regime]      = skewness(regime_returns)
        kurtosis[strategy][regime]  = kurtosis(regime_returns)

        # 2. Conditional Sharpe Ratio
        sharpe[strategy][regime] = E_return / vol  (annualized)

        # 3. Conditional Maximum Drawdown
        max_dd[strategy][regime] = max_drawdown(regime_returns)

        # 4. Conditional Alpha (Fung-Hsieh)
        alpha[strategy][regime] = regression(regime_returns ~ FH_7_factors)
        # Key question: Does alpha survive in all regimes or only some?

        # 5. Conditional Correlation with other strategies
        corr_matrix[regime] = correlation(all_strategy_returns[regime_labels == regime])
        # Key insight: diversification benefits change by regime
```

### 5.3 Expected Strategy-Regime Performance Map

Based on structural analysis and empirical evidence:

```
                    │  GROWTH    │  SLOWDOWN  │  CRISIS    │  RECOVERY  │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
L/S Equity          │  ★★★★      │  ★★        │  ★         │  ★★★       │
  (fundamental)     │  Long bias │  Dispersn↓ │  Corr spike│  Dispersn↑ │
                    │  helps     │  alpha hard│  beta pain │  alpha opp │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Global Macro        │  ★★★       │  ★★★★      │  ★★★       │  ★★        │
  (discretionary)   │  Carry +   │  Policy    │  Disc.     │  Crowded   │
                    │  trending  │  shifts    │  trades    │  recovery  │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
CTA / Managed       │  ★★★       │  ★★        │  ★★★★★     │  ★         │
Futures (trend)     │  Sustained │  Choppy    │  Strong    │  Trend     │
                    │  trends    │  reversals │  crisis α  │  reversals │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Event Driven        │  ★★★★      │  ★★        │  ★         │  ★★★       │
                    │  Deal flow │  Deals     │  Deals     │  Distressed│
                    │  high      │  slow      │  break     │  recovery  │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Relative Value      │  ★★★★      │  ★★★       │  ★         │  ★★★★      │
                    │  Spreads   │  Vol rising│  Spread    │  Spreads   │
                    │  tight,OK  │  but arb OK│  blowout   │  normalize │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Multi-Strategy      │  ★★★       │  ★★★       │  ★★        │  ★★★       │
                    │  Stable    │  Stable    │  Less bad  │  Stable    │
                    │  across    │  across    │  than pure │  across    │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Credit (L/S)        │  ★★★★      │  ★★        │  ★         │  ★★★★★     │
                    │  Carry     │  Spreads   │  Default   │  Spreads   │
                    │  accrual   │  widening  │  cycle     │  compress  │
────────────────────┼────────────┼────────────┼────────────┼────────────┤
Volatility          │  Depends   │  Depends   │  Long vol  │  Short vol │
                    │  Short vol │  Long vol  │  ★★★★★     │  ★★★★      │
                    │  ★★★★      │  ★★★       │  pays off  │  vol decay │
────────────────────┴────────────┴────────────┴────────────┴────────────┘
```

### 5.4 Transition Performance Analysis (CRITICAL and often overlooked)

The most valuable edge is understanding performance **during regime transitions**,
not just within regimes.

```
For each transition pair (e.g., Growth → Crisis):
  1. Identify transition months (month before + month of regime change)
  2. Compute strategy returns during transitions
  3. Key questions:
     - Which strategies protect during Growth→Crisis transition?
       (Answer: CTA, Long Vol, Global Macro)
     - Which strategies capture the Crisis→Recovery turn earliest?
       (Answer: Credit, Distressed, RV Arb)
     - Which strategies suffer most during Slowdown→Crisis?
       (Answer: Event Driven, L/S Equity with high beta)
     - Which strategies are indifferent to transitions?
       (Answer: Market Neutral, well-diversified Multi-Strat)
```

### 5.5 Regime-Conditional Correlation Matrix

This is one of the most important outputs. Correlation structure changes dramatically
across regimes:

```
Growth regime correlations (typical):
  - L/S Equity ↔ Event Driven: +0.5 (moderate)
  - CTA ↔ L/S Equity: ~0 (uncorrelated)
  - Credit ↔ L/S Equity: +0.3 (moderate)

Crisis regime correlations (typical):
  - L/S Equity ↔ Event Driven: +0.85 (very high — both equity-sensitive)
  - CTA ↔ L/S Equity: -0.4 (negative — CTA provides crisis alpha)
  - Credit ↔ L/S Equity: +0.7 (high — everything correlates in crisis)

KEY INSIGHT: Diversification you think you have in Growth disappears in Crisis.
The regime-conditional correlation matrix is essential for honest risk assessment.
```

---

## 6. Fund-Level Regime Analysis

### 6.1 Regime Beta

For each fund, estimate regime sensitivity:

```
R_fund_t = α + β_Growth · D_Growth_t + β_Slowdown · D_Slowdown_t
         + β_Crisis · D_Crisis_t + β_Recovery · D_Recovery_t + ε_t

Where D_k are regime dummy variables.

Output: a "regime beta profile" for each fund
  Fund A:  β_Growth=+0.8%, β_Slowdown=-0.2%, β_Crisis=-3.1%, β_Recovery=+1.5%
  Fund B:  β_Growth=+0.3%, β_Slowdown=+0.5%, β_Crisis=-0.5%, β_Recovery=+0.2%

Fund B is "all-weather" — Fund A is regime-sensitive
```

### 6.2 Regime-Conditional Alpha Decomposition

```
For each regime:
  R_fund_t = α_regime + Σ β_i · Factor_i_t + ε_t

  Using Fung-Hsieh 7 factors + supplementary factors

Key output:
  Fund A: α_Growth = +0.2% (p=0.05), α_Crisis = -0.8% (p=0.01)
  → This fund has positive alpha in Growth but NEGATIVE alpha in Crisis
  → Manager is adding value in calm markets but destroying it in stress

  Fund C: α_Growth = +0.1% (p=0.30), α_Crisis = +0.5% (p=0.03)
  → This fund has no significant alpha in Growth but significant POSITIVE crisis alpha
  → This is a valuable crisis diversifier
```

### 6.3 Style Drift Detection

```
Rolling 36-month factor regression:
  β_i(t) = factor loading at time t (rolling estimate)

Style drift score:
  Drift_t = Σ |β_i(t) - β_i(t-12)| / N_factors

Cross-reference with regime:
  - If drift correlates with regime → manager is adapting (potentially good)
  - If drift is random → manager is losing discipline (red flag)
  - If factor exposures change dramatically → reassess strategy classification
```

### 6.4 Fund Selection Score by Regime

For each fund, compute a composite score for each regime:

```
Score_fund_regime = w₁ · Sharpe_regime + w₂ · Alpha_regime + w₃ · (1 - MaxDD_regime/MaxDD_avg)
                  + w₄ · Consistency_regime

Where:
  Sharpe_regime = Sharpe ratio conditional on regime
  Alpha_regime = FH alpha conditional on regime (t-stat weighted)
  MaxDD_regime = max drawdown in this regime (normalized)
  Consistency_regime = % of months with positive return in this regime

This creates a fund×regime matrix that directly feeds the allocation engine.
```

---

## 7. Allocation Engine

### 7.1 Regime-Conditional Expected Returns

```
Step 1: Current regime probabilities from ensemble
  P = [P(Growth)=0.65, P(Slowdown)=0.25, P(Crisis)=0.05, P(Recovery)=0.05]

Step 2: Strategy expected returns conditional on each regime
  E[R_strategy | regime_k]  (from Section 5.2)

Step 3: Blended expected return
  E[R_strategy] = Σ_k P(regime_k) · E[R_strategy | regime_k]

Step 4: Forward-looking adjustment using transition matrix
  P_next = P · A   (one-step ahead regime probabilities)
  E[R_strategy, t+1] = Σ_k P_next(regime_k) · E[R_strategy | regime_k]

  Use weighted average of current and next-period:
  E_blended = 0.7 · E[R_strategy, t] + 0.3 · E[R_strategy, t+1]
```

### 7.2 Regime-Conditional Risk (Covariance)

```
Step 1: Covariance matrix per regime
  Σ_k = covariance(strategy_returns | regime_k)

Step 2: Blended covariance (accounts for within-regime AND between-regime risk)
  Σ = Σ_k P(regime_k) · Σ_k                           [within-regime]
    + Σ_k P(regime_k) · (μ_k - μ̄)(μ_k - μ̄)'          [between-regime]

  where μ_k = mean return vector in regime k, μ̄ = blended mean

This "total covariance" properly captures the regime-switching risk that
single-regime covariance matrices miss.
```

### 7.3 Allocation Methods (from simplest to most sophisticated)

#### Method A: Regime-Conditional Risk Parity (Recommended Baseline)

```
For each regime k:
  w_k = risk_parity_weights(Σ_k)   [equal risk contribution]

Blended allocation:
  w = Σ_k P(regime_k) · w_k

Advantages:
  - Doesn't require return forecasts (only covariance)
  - Naturally shifts to crisis-diversifying strategies when P(Crisis) rises
  - Robust to estimation error in expected returns
  - Good baseline to beat
```

#### Method B: Black-Litterman with Regime Views (Recommended Primary)

```
Step 1: Equilibrium returns (market-cap or equal-risk weighted)
  Π = δ · Σ · w_mkt

Step 2: Regime views
  Q = E[R_strategy | current regime]   (from Section 7.1)
  P = identity matrix (one view per strategy)
  Ω = uncertainty matrix:
    - High confidence regimes: Ω diagonal small
    - Low confidence / transition: Ω diagonal large

Step 3: Black-Litterman posterior
  E[R_BL] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹Π + P'Ω⁻¹Q]

Step 4: Optimize with posterior
  w = argmax  w'E[R_BL] - (λ/2)·w'Σw
  subject to: Σw = 1, w ≥ 0, w ≤ w_max

Key insight: When regime confidence is LOW (Ω large), BL falls back to equilibrium.
When confidence is HIGH (Ω small), BL leans into regime views.
This naturally produces more conservative allocations during transitions.
```

#### Method C: Robust Mean-CVaR (Most Sophisticated)

```
For each regime k, estimate:
  - CVaR_α(k) = expected loss beyond α-quantile (typically α=5%)
  - Computed from regime-conditional return distribution

Optimization:
  w = argmax  w'E[R] - λ · CVaR_α(w, regime_probs)

  where CVaR is computed over the regime-mixture distribution:
    R_portfolio ~ Σ_k P(k) · N(μ_k, Σ_k)  [regime mixture]

Advantages:
  - Explicitly accounts for tail risk in crisis regimes
  - Regime-mixture distribution has fatter tails than any single regime
  - More appropriate for FoF where tail risk management is paramount
```

### 7.4 Confidence Scaling

```
Base allocation: w_base (from Method A, B, or C above)

Confidence adjustment:
  If confidence = HIGH (models agree):
    w_final = w_base  (full conviction)

  If confidence = LOW (models disagree):
    w_final = (1-α) · w_neutral + α · w_base
    where w_neutral = equal-weight or risk-parity (no regime view)
    and α = confidence_score ∈ [0.3, 0.7]

  If BOCPD alert = Red/Emergency:
    w_final = w_defensive
    where w_defensive = increased CTA + Long Vol + Macro, reduced Credit + ED

This prevents the system from making large bets during uncertain transitions.
```

### 7.5 Constraints & Overlays

```
Practical constraints for a liquid alts FoF:

1. LIQUIDITY CONSTRAINTS
   - Max allocation to any fund ≤ X% of fund's AUM (avoid being too large in a fund)
   - Weight funds with quarterly redemption terms lower than monthly liquidity funds
   - Maintain minimum cash/liquid buffer for redemptions

2. CONCENTRATION LIMITS
   - Max single fund: 10-15% of portfolio
   - Max single strategy cluster: 30-40% of portfolio
   - Min strategy clusters: ≥ 4 (diversification floor)

3. TURNOVER LIMITS
   - Max monthly rebalance: 20-30% of portfolio
   - Smooth transitions: exponential smoothing of target weights
     w_target_smooth = γ · w_new + (1-γ) · w_previous   (γ ≈ 0.3-0.5)
   - This prevents whipsaw from regime oscillations

4. RISK BUDGET
   - Max contribution to portfolio vol per strategy: 30%
   - Max portfolio vol (ex-ante): calibrated to mandate
   - Regime-conditional VaR budget

5. REDEMPTION AWARENESS
   - If fund has 45-day notice + quarter-end redemption:
     cannot reduce allocation faster than notice period allows
   - Model "executable allocation" vs "target allocation"
   - Maintain liquidity ladder
```

### 7.6 Dynamic Rebalancing Triggers

```
MONTHLY (standard cycle):
  1. Update all features and signals
  2. Run Jump Model + HMM ensemble
  3. Compute new regime probabilities
  4. Generate target allocation
  5. If |w_target - w_current| > threshold → rebalance
  6. Submit redemption/subscription notices as needed

INTRA-MONTH (event-driven):
  Trigger 1: BOCPD Red Alert
    → Review regime, but don't automatically trade
    → Assess: is this a real regime change or noise?
    → If confirmed by ≥3 signals: accelerate rebalance timeline

  Trigger 2: Single-fund drawdown > X%
    → Fund-level review (not regime-level)
    → May reduce/redeem regardless of regime

  Trigger 3: Portfolio vol exceeds risk budget
    → De-risk pro-rata or reduce highest-vol positions
    → Override regime allocation if necessary
```

---

## 8. Monitoring & Alert System

### 8.1 Daily Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  REGIME ENGINE DASHBOARD — 2026-02-13                       │
│                                                             │
│  CURRENT REGIME: Growth (72% confidence)                    │
│  Regime Age: 7 months                                       │
│  BOCPD Status: ● Green (no alerts)                         │
│                                                             │
│  ┌─── SIGNAL HEATMAP ──────────────────────────────────┐   │
│  │  Absorption Ratio:    ▓▓▓▓░░░░░░  42nd %ile  ● OK  │   │
│  │  ΔAR (1M):            ▓▓▓░░░░░░░  28th %ile  ● OK  │   │
│  │  ENB:                 ▓▓▓▓▓▓░░░░  61st %ile  ● OK  │   │
│  │  Turbulence:          ▓▓▓░░░░░░░  33rd %ile  ● OK  │   │
│  │  Corr Surprise:       ▓▓▓▓░░░░░░  45th %ile  ● OK  │   │
│  │  VIX %ile:            ▓▓░░░░░░░░  22nd %ile  ● OK  │   │
│  │  VIX Term Structure:  ▓▓▓▓▓▓▓░░░  0.89       ● OK  │   │
│  │  HY OAS %ile:         ▓▓▓░░░░░░░  35th %ile  ● OK  │   │
│  │  Credit Momentum:     ▓▓▓▓░░░░░░  40th %ile  ● OK  │   │
│  │  Yield Curve (10-2):  ▓▓▓▓▓░░░░░  +52bp      ● OK  │   │
│  │  NFCI:                ▓▓▓░░░░░░░  -0.35      ● OK  │   │
│  │  Stock-Bond Corr:     ▓▓▓▓▓▓░░░░  -0.22      ● OK  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  REGIME PROBABILITIES:                                      │
│    Growth:   ▓▓▓▓▓▓▓░░░  72%                               │
│    Slowdown: ▓▓░░░░░░░░  18%                               │
│    Crisis:   ░░░░░░░░░░   3%                               │
│    Recovery: ▓░░░░░░░░░   7%                               │
│                                                             │
│  MODELS:                                                    │
│    Jump Model:  Growth  (distance to centroid: 1.2)         │
│    HMM:         Growth  (filtered prob: 0.74)               │
│    Agreement:   ✓ YES   Confidence: HIGH                    │
│                                                             │
│  BOCPD CHANGEPOINT PROBABILITIES (5-day avg):               │
│    VIX:           2%  ●                                     │
│    HY Spreads:    4%  ●                                     │
│    Absorption:    3%  ●                                     │
│    SB Corr:       8%  ●                                     │
│    NFCI:          1%  ●                                     │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Monthly Report Components

1. **Regime assessment**: Current regime, probabilities, model agreement, confidence
2. **Signal evolution**: How each signal moved over the past month (table + sparklines)
3. **Transition risk**: HMM transition matrix → P(regime change in next 1, 3, 6 months)
4. **Strategy performance**: MTD/QTD/YTD by strategy, with regime context
5. **Allocation recommendation**: Target weights vs current, required trades
6. **Historical regime timeline**: Annotated timeline with market events
7. **Risk decomposition**: Portfolio risk contribution by strategy, by regime scenario

### 8.3 Alert Definitions

| Alert Level | Trigger | Action |
|-------------|---------|--------|
| **Green** | All signals normal, models agree | Standard monthly cycle |
| **Yellow** | BOCPD P(CP) > 0.7 on 1 signal, OR model disagreement | Monitor daily, prepare contingency |
| **Orange** | BOCPD P(CP) > 0.7 on 2+ signals, OR ΔAR > +2σ | Intra-month review, reduce leverage |
| **Red** | BOCPD P(CP) > 0.9 on 3+ signals, OR VIX > 95th %ile | Emergency review, activate defensive allocation |

---

## 9. Backtesting Framework

### 9.1 Methodology

```
EXPANDING WINDOW BACKTEST:

For t = 60 to T (months):
  1. Train Jump Model on data [0, t]
  2. Train HMM on data [0, t]
  3. Generate regime probabilities at time t
  4. Compute target allocation at time t
  5. Record out-of-sample return at t+1

No look-ahead anywhere:
  - Features computed on expanding window up to t
  - Models trained on data up to t
  - Allocation decided at t, return measured at t+1
  - Even percentile rank standardization uses only [0, t] data
```

### 9.2 Benchmarks to Beat

| Benchmark | Description |
|-----------|-------------|
| **Equal Weight** | 1/N across all strategy indices |
| **Equal Risk** | Static risk parity across strategies |
| **60/40 Liquid Alts** | 60% equity-oriented (L/S, ED), 40% diversifiers (Macro, CTA, RV) |
| **Vol Target** | Equal-weight with vol scaling (target 6-8% annualized) |
| **HMM Only** | Single HMM model, no ensemble |
| **Buy Best Sharpe** | Always allocate to highest trailing Sharpe strategy |
| **HFRI FoF Composite** | Industry benchmark for fund of funds |

### 9.3 Key Metrics

```
RETURN METRICS:
  - Annualized return
  - Annualized vol
  - Sharpe ratio
  - Sortino ratio

RISK METRICS:
  - Maximum drawdown
  - Calmar ratio (return / max DD)
  - CVaR (5%)
  - Worst month / worst quarter

REGIME-SPECIFIC METRICS:
  - Sharpe by regime (is the system adding value in each regime?)
  - Crisis alpha: return during Crisis regime months
  - Transition performance: return in regime-change months
  - Regime detection accuracy: % of months where regime label matches
    ex-post "true" regime (defined by subsequent 3-month returns)
  - Detection latency: average months delay in detecting regime change

IMPLEMENTATION METRICS:
  - Turnover (monthly)
  - Strategy concentration (Herfindahl of weights)
  - Number of regime changes per year
  - Time in each regime
```

### 9.4 Robustness Tests

1. **Regime count sensitivity**: Repeat with K=3 and K=5 states
2. **Feature subset stability**: Remove one feature category at a time
3. **Parameter sensitivity**: Vary jump penalty λ, HMM priors, BOCPD hazard
4. **Walk-forward stability**: Are regime labels consistent across retraining windows?
5. **Crisis stress test**: Specifically evaluate 2008, 2011, 2015, 2018Q4, 2020 COVID, 2022
6. **Turnover constraint sensitivity**: How much value is lost with stricter turnover limits?
7. **Lag sensitivity**: What if fund returns are reported with 1-3 month lag?

---

## 10. Build Phases

### Phase 1: Data Infrastructure + Feature Engineering (Weeks 1-3)

```
Deliverables:
  ✦ Bloomberg data pipeline (daily market + monthly macro)
  ✦ Fund return database loader (handle mixed frequencies)
  ✦ HF index data loader (HFRI, CS, SG)
  ✦ All 26 features computed and validated
  ✦ Feature correlation analysis (identify redundancies)
  ✦ Historical feature visualization with crisis overlays

Notebooks: 01_data_exploration.ipynb, 02_feature_signals.ipynb
```

### Phase 2: Global Regime Model (Weeks 3-5)

```
Deliverables:
  ✦ Statistical Jump Model implementation + calibration
  ✦ Bayesian HMM implementation + calibration
  ✦ BOCPD implementation on daily signals
  ✦ Ensemble logic
  ✦ Historical regime timeline (validate against known events)
  ✦ Regime profiles (what defines each regime)
  ✦ Feature importance analysis (sparse JM)

Notebook: 03_global_regime_model.ipynb
```

### Phase 3: Strategy & Fund Mapping (Weeks 5-7)

```
Deliverables:
  ✦ Strategy conditional performance tables
  ✦ Regime-conditional correlation matrices
  ✦ Transition performance analysis
  ✦ Fung-Hsieh alpha decomposition by regime
  ✦ Fund-level regime betas
  ✦ Fund-level regime-conditional alpha
  ✦ Style drift detection
  ✦ Fund selection scores by regime

Notebooks: 04_strategy_sub_regimes.ipynb, 05_fund_regime_mapping.ipynb
```

### Phase 4: Allocation Engine + Backtest (Weeks 7-10)

```
Deliverables:
  ✦ Regime-conditional risk parity (baseline)
  ✦ Black-Litterman with regime views (primary)
  ✦ Confidence scaling logic
  ✦ Constraint implementation (liquidity, concentration, turnover)
  ✦ Full expanding-window backtest
  ✦ Benchmark comparison
  ✦ Robustness tests
  ✦ Dashboard prototype

Notebook: 06_allocation_backtest.ipynb
```

### Phase 5: Production & Monitoring (Weeks 10-12)

```
Deliverables:
  ✦ Refactor notebooks into clean Python modules
  ✦ Config-driven pipeline (YAML)
  ✦ Monthly regime report generation
  ✦ Daily signal monitoring script
  ✦ Alert system
  ✦ Documentation
```

---

## Appendix A: Key References

| Paper / Resource | Authors | Year | Relevance |
|-----------------|---------|------|-----------|
| Principal Components as a Measure of Systemic Risk | Kritzman, Li, Page, Rigobon | 2010 | Absorption Ratio |
| Managing Diversification | Meucci | 2009 | ENB / Effective Number of Bets |
| Correlation Surprise | Kinlaw, Turkington | 2012 | Turbulence decomposition |
| Regime Changes and Financial Markets | Ang, Timmermann | 2012 | HMM for finance (survey) |
| Learning HMMs with Persistent States by Penalizing Jumps | Nystrup et al. | 2020 | Statistical Jump Model |
| Downside Risk Reduction: A Statistical Jump Model Approach | Shu, Mulvey | 2024 | Jump Model for allocation |
| Feature Selection in Jump Models | Nystrup et al. | 2021 | Sparse Jump Model |
| Bayesian Online Changepoint Detection | Adams, MacKay | 2007 | BOCPD |
| Asset Allocation and the Markov Regime-Switching Model | Guidolin, Timmermann | 2007 | Regime-switching allocation |
| Hedge Fund Replication Using Strategy-Specific Factors | Fung, Hsieh | 2004 | 7-Factor Model |

## Appendix B: Python Dependencies

```
# Core
numpy, pandas, scipy, scikit-learn

# Models
jump-models          # Statistical Jump Model (Shu)
hmmlearn             # HMM (Gaussian, can extend to Student-t)
pomegranate          # Alternative HMM with more distributions
pymc                 # Bayesian HMM (full posterior)

# Features
arch                 # GARCH, realized volatility
statsmodels          # Time series, regressions

# Bloomberg
xbbg                 # Bloomberg Python API wrapper
pdblp                # Alternative Bloomberg wrapper

# Visualization
matplotlib, seaborn, plotly

# Optimization
cvxpy                # Convex optimization (for allocation)
riskfolio-lib        # Portfolio optimization with risk measures
```
