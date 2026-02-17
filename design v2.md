# DESIGN V2 — Critical Upgrades for Institutional-Grade System
# 8 Missing Components That Separate Good from Best-of-the-Best

The original DESIGN.md is strong on signals and model ensemble. But after stress-testing
against what Two Sigma, SSGA, JP Morgan/GIC, and Man FRM actually deploy, **8 critical
components were missing**. This document defines them.

Read this as a mandatory supplement to DESIGN.md. Each section below slots into the
original architecture at a specific layer.

---

## Table of Contents

1. [What Was Missing and Why It Matters](#1-gap-analysis)
2. [Return Unsmoothing (Getmansky-Lo-Makarov)](#2-return-unsmoothing)
3. [Hedge Fund Nowcasting](#3-hedge-fund-nowcasting)
4. [Factor Crowding Detection](#4-factor-crowding-detection)
5. [Implied Correlation Signal](#5-implied-correlation-signal)
6. [Contagion / Spillover Framework (Diebold-Yilmaz)](#6-contagion-spillover)
7. [Alternative Risk Premia Decomposition](#7-alternative-risk-premia)
8. [Regime-Switching Factor Model](#8-regime-switching-factor-model)
9. [Drawdown Regime Overlay](#9-drawdown-regime-overlay)
10. [Functional Fund Classification (JP Morgan/GIC)](#10-functional-classification)
11. [Institutional Validation — What the Best Actually Use](#11-institutional-validation)
12. [Updated Architecture Diagram](#12-updated-architecture)
13. [Updated Feature Matrix (32 Signals)](#13-updated-feature-matrix)
14. [Updated Build Phases](#14-updated-build-phases)
15. [Complete Reference Library](#15-references)

---

## 1. Gap Analysis — What Was Missing and Why It Matters

| # | Missing Component | Why It's Critical for Liquid Alts FoF | Consequence of Omission |
|---|------------------|--------------------------------------|------------------------|
| 1 | **Return Unsmoothing** | HF returns are smoothed by illiquidity/stale pricing → artificial serial correlation | Regime signals delayed 1-3 months. You detect crises AFTER they happen. |
| 2 | **Nowcasting** | Monthly-reporting funds leave you blind for 30-45 days | Cannot do dynamic/event-driven rebalancing with monthly data alone |
| 3 | **Factor Crowding** | Aug 2007 quant crisis was invisible to ALL macro regime models | Entire class of regime risk (crowding unwinds) is undetectable |
| 4 | **Implied Correlation** | Forward-looking signal from options market | Missing 1-3 week lead on correlation regime shifts |
| 5 | **Contagion Framework** | Stress propagates in a specific order across strategies | Without this, you don't know what to exit first in a crisis |
| 6 | **ARP Decomposition** | Every top allocator separates traditional beta / alt beta / alpha | Cannot distinguish fee-worthy alpha from replicable beta exposure |
| 7 | **Regime-Switching Factors** | Factor betas change dramatically across regimes (Billio et al.) | Constant-beta models give WRONG risk estimates in crisis |
| 8 | **Drawdown Overlay** | Funds are redeemed on drawdown thresholds, not on macro signals | No systematic drawdown management → the #1 cause of FoF failure |

**Bottom line**: The original design had the right core engine (Jump Model + HMM + BOCPD
ensemble with 26 signals). But it was treating hedge fund returns as clean data (they're
not), missing an entire regime dimension (crowding), lacking forward-looking signals
(implied correlation), had no real-time visibility (nowcasting), and missed the risk
management layer that actually determines survival (drawdown overlay + contagion ordering).

---

## 2. Return Unsmoothing — Getmansky, Lo, Makarov (2004)

**Slots into**: Data Layer, before any feature computation or model training

### The Problem

Hedge fund reported returns exhibit artificial serial correlation due to:
- Illiquid holdings priced with stale marks
- Monthly NAV computation averaging intra-month prices
- Manager discretion in marking OTC positions
- Administrative lag in collecting fund valuations

This smoothing DELAYS regime signals. If a fund is truly down -5% in month t but reports
-2% in month t and -3% in month t+1, your regime model sees the damage late.

### The Model

Getmansky, Lo, Makarov model observed (reported) returns as a moving average of true
(economic) returns:

```
R_observed_t = θ₀ · R_true_t + θ₁ · R_true_{t-1} + θ₂ · R_true_{t-2}

Constraints:
  θ₀ + θ₁ + θ₂ = 1     (returns sum correctly)
  θ_j ∈ [0, 1]          (no negative weights)
  θ₀ ≥ θ₁ ≥ θ₂          (most recent return has highest weight)
```

### Estimation

```
Step 1: Estimate MA(2) coefficients via MLE
  Fit: R_observed_t = θ₀ · ε_t + θ₁ · ε_{t-1} + θ₂ · ε_{t-2}
  where ε_t ~ N(0, σ²) are the true (unobserved) returns

  Use statsmodels ARMA(0,2) with constrained optimization

Step 2: Compute Smoothing Index
  ξ = θ₀² + θ₁² + θ₂²    (Herfindahl of theta weights)

  ξ = 1.0  →  no smoothing (θ₀=1, θ₁=θ₂=0)
  ξ = 0.33 →  maximum smoothing (θ₀=θ₁=θ₂=1/3)

Step 3: Invert to recover true returns
  Using z-transform inversion:

  R_true_t = (R_observed_t - θ₁ · R_true_{t-1} - θ₂ · R_true_{t-2}) / θ₀
```

### Practical Impact by Strategy

| Strategy | Typical ξ (Smoothing Index) | Typical θ₀ | Action |
|----------|-----------------------------|-----------|--------|
| Equity L/S | 0.85-0.95 | 0.90+ | Minimal unsmoothing needed |
| Global Macro | 0.80-0.90 | 0.85+ | Light unsmoothing |
| CTA/Managed Futures | 0.90-1.00 | 0.95+ | Nearly no smoothing (daily-priced) |
| Event Driven | 0.70-0.85 | 0.80 | Moderate unsmoothing |
| Relative Value | 0.55-0.75 | 0.70 | **Significant unsmoothing needed** |
| Credit / Structured | 0.45-0.65 | 0.60 | **Heavy unsmoothing needed** |
| Convertible Arb | 0.50-0.70 | 0.65 | **Heavy unsmoothing needed** |

### Implementation Rules

1. **Estimate θ per fund** using full available history (min 36 months)
2. **Re-estimate annually** (smoothing characteristics can change)
3. **Apply unsmoothing to all fund returns** before regime model training
4. **Apply to strategy index returns** (HFRI etc.) — they are also smoothed
5. **Unsmoothed returns have higher vol and lower Sharpe** — this is the TRUE risk
6. **Unsmoothed correlations are higher** — true diversification is less than reported
7. **Regime model on unsmoothed data detects transitions 1-2 months earlier**

### Reference
- Getmansky, Lo, Makarov (2004). "An Econometric Model of Serial Correlation and
  Illiquidity in Hedge Fund Returns." Journal of Financial Economics, 74(3), 529-609.

---

## 3. Hedge Fund Nowcasting — Daily Performance Estimation

**Slots into**: Data Layer + Monitoring Layer

### The Problem

Your regime model can detect a crisis in real-time (daily BOCPD, daily signals). But you
don't know how your PORTFOLIO is performing until fund NAVs arrive 15-45 days later. This
makes dynamic rebalancing impossible in practice.

### Solution: Factor-Based Nowcasting (Kalman Filter DSA)

```
MONTHLY CALIBRATION (expanding window):

For each fund, estimate factor loadings via Dynamic Style Analysis:

  R_fund_t = Σᵢ βᵢ,t · Fᵢ,t + εt

  where F = daily-tradeable factor proxies:
    1. SPY  (equity market)
    2. IWM  (small cap)
    3. EFA  (intl developed)
    4. EEM  (EM equity)
    5. TLT  (long duration bonds)
    6. LQD  (IG credit)
    7. HYG  (HY credit)
    8. GLD  (gold)
    9. DBC  (commodities)
    10. UUP  (US dollar)
    11. SHV  (cash / risk-free)

  Constraints:
    Σ βᵢ = 1    (fully invested)
    βᵢ ≥ 0      (long-only factor exposures)

  Estimation method: Kalman Filter (preferred) or 24-month rolling OLS
    - Kalman allows βᵢ to evolve smoothly over time
    - Captures manager's changing exposures

DAILY PROJECTION:

  R̂_fund_d = Σᵢ β̂ᵢ,T · Fᵢ,d

  where T = last monthly calibration, d = today

MONTH-TO-DATE NOWCAST:

  R̂_fund_MTD = Π(1 + R̂_fund_d) - 1  for d in current month
```

### Nowcast Quality by Strategy (Expected R²)

| Strategy | Typical R² | Quality | Reason |
|----------|-----------|---------|--------|
| L/S Equity (high net) | 0.80-0.90 | Excellent | Dominated by equity beta |
| L/S Equity (low net) | 0.40-0.60 | Moderate | Alpha-driven, harder to replicate |
| Global Macro | 0.50-0.70 | Moderate | Multiple asset classes, directional |
| CTA / Trend | 0.60-0.80 | Good | Trend factors replicate well |
| Event Driven | 0.30-0.50 | Weak | Deal-specific, idiosyncratic |
| Relative Value | 0.20-0.40 | Weak | Spread-driven, not in proxy universe |
| Multi-Strategy | 0.40-0.60 | Moderate | Blended exposure |

### How Nowcasting Enables Dynamic Rebalancing

```
SCENARIO: BOCPD fires Red Alert on day 12 of the month

WITHOUT NOWCASTING:
  - You know the regime is changing
  - You DON'T know how your funds are performing
  - Cannot assess portfolio damage or decide what to cut
  - Must wait 18-48 more days for actual NAVs

WITH NOWCASTING:
  - BOCPD fires Red Alert
  - Instantly compute: estimated MTD return for each fund
  - Identify: which funds are likely already impaired
  - Assess: portfolio-level estimated drawdown
  - Decide: submit redemption notices for most-impaired / highest-risk funds
  - Timeline: react within 24 hours instead of 30-45 days
```

### Implementation

```python
# Core approach: Kalman Filter Dynamic Style Analysis
from pykalman import KalmanFilter

# State: factor loadings β (11-dimensional)
# Observation: fund monthly return
# Transition: β_t = β_{t-1} + η_t  (random walk on betas)
# Observation: R_t = β_t · F_t + ε_t

kf = KalmanFilter(
    transition_matrices=np.eye(11),           # random walk
    observation_matrices=F_monthly,            # factor returns matrix
    transition_covariance=0.01 * np.eye(11),   # slow-evolving betas
    observation_covariance=obs_var,             # fund-specific noise
    initial_state_mean=ols_betas,              # initialize from OLS
    initial_state_covariance=np.eye(11)
)

# Filter: estimate betas up to each month
filtered_betas, _ = kf.filter(fund_returns)

# Nowcast: use latest betas with daily factor returns
daily_nowcast = daily_factors @ filtered_betas[-1]
```

### Reference
- Wermers (2014). "Monitoring Daily Hedge Fund Performance."
  Journal of Investment Consulting, 15(1).

---

## 4. Factor Crowding Detection

**Slots into**: Feature Engineering (new Signal 27) + Separate Regime Dimension

### Why This Is Critical

The August 2007 quant crisis destroyed billions in value across quantitative hedge funds
in a matter of days. **Every macro indicator was benign.** VIX was normal. Credit spreads
were tight. The yield curve was fine. No macro regime model would have detected it.

The cause was **factor crowding** — too many managers holding the same positions. When one
large fund was forced to deleverage (reportedly Goldman Sachs' Global Alpha), the selling
cascaded through every fund holding similar positions.

This is a distinct regime dimension that macro models are structurally blind to.

### Measuring Crowding: The CoMetric (Baltas, 2019)

```
For a given hedge fund strategy / factor:

Step 1: Compute factor-adjusted residual returns
  For each fund i in the strategy:
    ε_i,t = R_i,t - Σ β_k · Factor_k,t

  (Strip out market factors to isolate strategy-specific behavior)

Step 2: Compute pairwise correlations of residuals
  ρ_ij,t = corr(ε_i, ε_j) over trailing 52 weeks

Step 3: CoMetric = average pairwise residual correlation
  CoMetric_t = (2 / (N(N-1))) · Σ_{i<j} ρ_ij,t

  where N = number of funds in the strategy cluster

Step 4: Classify crowding regime
  CoMetric > 80th percentile of history → "Crowded"
  CoMetric < 40th percentile → "Uncrowded"
  Between → "Moderate"
```

### Alternative Crowding Measures

```
1. 13F OVERLAP SCORE (if holdings data available):
   Overlap_ij = 2 · Σ min(w_i_k, w_j_k) / (Σw_i_k + Σw_j_k)
   Average across fund pairs → portfolio crowding score

2. SHORT INTEREST CONCENTRATION:
   For equity L/S: aggregate short interest in the most-shorted names
   Spikes in concentrated short interest = crowded short positions

3. FACTOR RETURN AUTOCORRELATION:
   When a factor exhibits negative autocorrelation (reversal),
   it may indicate crowded positions being unwound.
   Autocorr_factor_t = corr(F_t, F_{t-1}) over rolling 60 days
   Persistent negative autocorrelation = crowding unwind signal

4. DISPERSION OF FACTOR LOADINGS:
   Within a strategy cluster, compute σ(β_i) across funds
   LOW dispersion = everyone doing the same thing = crowded
   HIGH dispersion = differentiated = uncrowded
```

### Crowding Regime Interaction with Macro Regime

```
                           MACRO REGIME
                     Growth        Crisis
                  ┌───────────┬───────────┐
CROWDING    Low   │  IDEAL    │  PAINFUL  │
REGIME            │  Alpha    │  but      │
                  │  harvest  │  survivable│
                  ├───────────┼───────────┤
            High  │  FRAGILE  │  CATASTROPHIC│
                  │  Looks    │  Aug 2007  │
                  │  good but │  scenario  │
                  │  vulnerable│           │
                  └───────────┴───────────┘

KEY INSIGHT: High Crowding + Growth is the MOST DANGEROUS regime
because it looks fine but is maximally fragile. This is Two Sigma's
"Walking on Ice" regime.
```

### Implementation in the System

- Compute CoMetric for each strategy cluster monthly
- Add as Signal 27 in the feature matrix
- Also maintain as a **separate binary regime overlay** (Crowded / Uncrowded)
- When Crowded + any BOCPD alert → highest priority warning

### References
- Khandani & Lo (2007). "What Happened to the Quants in August 2007?"
- Baltas (2019). "The Impact of Crowding in Alternative Risk Premia Investing."
  Financial Analysts Journal, 75(3).

---

## 5. Implied Correlation Signal

**Slots into**: Feature Engineering (new Signal 28)

### What It Is

The CBOE Implied Correlation Index measures the expected future correlation between
S&P 500 constituents, derived from the options market. It captures information that
realized correlation cannot — specifically, the **forward-looking expectations** of
option market participants.

### Key Empirical Finding: Correlation Risk Premium (CRP)

```
CRP = Implied Correlation - Realized Correlation

Empirical averages:
  S&P 500:  Implied ~39.5%, Realized ~32.5%  →  CRP ≈ +7%
  DJIA:     Implied ~46.0%, Realized ~35.5%  →  CRP ≈ +10.5%
```

This premium exists because investors pay for index-level (correlation) protection.
When the premium collapses or inverts, it signals regime stress.

### Signal Construction

```
Signal 28a: Implied Correlation Percentile
  IC_pctile = percentile_rank(IC_t, expanding window)

  IC > 80th %ile → markets expect high correlation → Crisis/Stress
  IC < 30th %ile → markets expect dispersion → Growth/Alpha regime

Signal 28b: Correlation Risk Premium (CRP) Gap
  CRP_t = Implied_Corr_t - Realized_Corr_60d_t
  CRP_pctile = percentile_rank(CRP_t, expanding)

  CRP widening (implied >> realized) → market PRICING IN future stress
                                       even though realized is still calm
                                    → LEADING indicator (1-3 weeks ahead)

  CRP collapsing (implied ≈ realized) → stress already HERE
                                       → COINCIDENT indicator

  CRP inverting (implied < realized)  → panic, extreme stress
                                       → CONTRARIAN buy signal
```

### Why This Is Superior to Realized Correlation for Regime Detection

| Property | Realized Correlation | Implied Correlation |
|----------|---------------------|---------------------|
| Timing | Backward-looking (trailing 60d) | Forward-looking (next 30-365d) |
| Regime lead | Lags transitions by weeks | Leads transitions by 1-3 weeks |
| Information source | Historical returns | Option market pricing |
| Noise | High (realized is noisy) | Lower (aggregated market expectations) |
| Crisis signal | Fires during/after crisis | Fires BEFORE crisis |

### Bloomberg Tickers
- ICJ Index (CBOE Implied Correlation, 1Y)
- KCJ Index (CBOE Implied Correlation, 2Y)
- COR1M Index, COR3M Index (shorter term)

### Reference
- Driessen, Maenhout, Vilkov (2009). "The Price of Correlation Risk."
  Journal of Finance, 64(3).

---

## 6. Contagion / Spillover Framework — Diebold-Yilmaz Connectedness

**Slots into**: Feature Engineering (new Signal 29) + Application Layer (exit ordering)

### Why This Matters for FoF

During a crisis, you need to know:
1. **Which strategies will crack first** (so you can exit early)
2. **Which strategies transmit stress to others** (net transmitters)
3. **Which strategies are insulated** (net receivers or disconnected)
4. **How connected the overall system is** (total connectedness as regime signal)

### The Diebold-Yilmaz Framework

```
Step 1: Estimate VAR(p) on strategy index returns
  Y_t = [R_LS_Eq, R_Macro, R_CTA, R_ED, R_RV, R_Credit, R_Vol]'
  Y_t = A₁Y_{t-1} + ... + ApY_{t-p} + u_t

  Use p=2 or BIC-selected lag order
  Estimate on rolling 200-day window

Step 2: Generalized Forecast Error Variance Decomposition (GFEVD)
  Compute H-step ahead (H=10 days) FEVD using Pesaran-Shin identification:

  d_ij(H) = variance of strategy i explained by shocks to strategy j

  (Pesaran-Shin is order-invariant, unlike Cholesky decomposition)

Step 3: Normalize
  d̃_ij = d_ij / Σⱼ d_ij    (row-normalize so each row sums to 1)

Step 4: Compute Connectedness Measures

  TOTAL CONNECTEDNESS (system-wide regime signal):
    C = (1/N) · Σ_{i≠j} d̃_ij × 100

    C = 40-50% → normal connectedness → Growth
    C = 60-70% → elevated → Slowdown
    C > 70%    → extreme → Crisis (everything moving together)

  FROM CONNECTEDNESS (how exposed is strategy i to all others):
    C_i← = Σ_{j≠i} d̃_ij × 100

    High C_i← = strategy i is highly vulnerable to cross-strategy shocks

  TO CONNECTEDNESS (how much does strategy i transmit to others):
    C_→i = Σ_{j≠i} d̃_ji × 100

    High C_→i = strategy i is a systemic risk transmitter

  NET CONNECTEDNESS:
    C_i_net = C_→i - C_i←

    Positive = net transmitter (source of contagion)
    Negative = net receiver (victim of contagion)
```

### Empirical Contagion Hierarchy (Crisis Propagation Order)

Based on Billio-Getmansky-Lo-Pelizzon (2012) and Boyson-Stahel-Stulz (2010):

```
TYPICAL CRISIS PROPAGATION SEQUENCE:

  Stage 1 (FIRST TO CRACK):
    → Convertible Arb / Fixed Income Arb
    → Leveraged credit strategies
    (Driven by: funding stress, margin calls, spread blowout)

  Stage 2 (CONTAGION SPREADS):
    → Event Driven / Multi-Strategy
    → Statistical Arbitrage
    (Driven by: forced deleveraging, prime broker risk, liquidity withdrawal)

  Stage 3 (BROAD IMPACT):
    → Equity Long/Short
    → Distressed Credit
    (Driven by: market-wide selling, beta exposure, redemptions)

  Stage 4 (LAST TO BE AFFECTED):
    → Global Macro (discretionary)
    → CTA / Managed Futures
    (Often POSITIVE in this stage — "crisis alpha")

  IMPLICATION FOR EXIT ORDERING:
    In a crisis, exit/reduce in REVERSE order of vulnerability:
    First reduce: RV/Arb, Credit → most fragile
    Then reduce: Event Driven → moderate fragility
    Hold/increase: CTA, Macro → crisis diversifiers
```

### Prime Broker Channel (Chung & Kang, 2016)

Strong co-movement exists among hedge funds sharing the same prime broker. Shocks
transmit through the prime brokerage network. If your FoF has multiple funds at the
same PB, you have concentrated PB contagion risk.

```
PB_Concentration = Herfindahl of AUM across prime brokers
High PB_Concentration → elevated contagion risk
```

### Signal Construction

```
Signal 29a: Total Connectedness Index (rolling 200-day)
  TCI_t = C(H=10) computed on strategy returns
  TCI_pctile = percentile_rank(TCI_t, expanding)

  TCI > 70th %ile → elevated systemic connectedness → Crisis precursor

Signal 29b: Net Connectedness Vector (directional)
  NC_strategy_t = C_→i - C_i←  for each strategy

  Use to identify which strategies are currently net transmitters
  (early warning of where stress originates)
```

### References
- Diebold & Yilmaz (2014). "On the Network Topology of Variance Decompositions."
  Journal of Econometrics, 182(1).
- Billio, Getmansky, Lo, Pelizzon (2012). "Econometric Measures of Connectedness
  and Systemic Risk in the Finance and Insurance Sectors." Journal of Financial
  Economics, 104(3), 535-559.
- Boyson, Stahel, Stulz (2010). "Hedge Fund Contagion and Liquidity Shocks."
  Journal of Finance, 65(5).

---

## 7. Alternative Risk Premia Decomposition

**Slots into**: Application Layer, between Strategy-Regime Mapping and Allocation Engine

### Why Every Top Allocator Does This

Research (Ardia, Barras, Gagliardini, Scaillet, 2024 JFE) shows that under standard
models, hedge fund alpha appears to be ~2.7% p.a. (positive for >70% of funds). Under
comprehensive factor models, alpha drops to ~0.4% p.a. and beta represents **93% of
average hedge fund returns**.

If you're paying 2-and-20 for replicable factor exposure, you're overpaying by 10x.
The regime model should tell you: given the current regime, how much of each strategy's
expected return is alpha vs. replicable beta?

### The Three-Layer Decomposition

```
For each fund (or strategy index):

R_fund = Traditional Beta + Alternative Beta + Pure Alpha

LAYER 1 — TRADITIONAL BETA:
  Exposure to long-only markets that any index fund provides:
  β_SPX · R_SPX + β_AGG · R_AGG + β_CREDIT · R_Credit

  This should be FREE (index funds, ETFs)

LAYER 2 — ALTERNATIVE BETA (Alternative Risk Premia):
  Systematic, rule-based strategies requiring non-traditional techniques
  (shorting, leverage, derivatives) but that are replicable:

  | ARP Factor   | Description                              | Proxy                    |
  |-------------|------------------------------------------|--------------------------|
  | Value       | Buy cheap / sell expensive                | HML, AQR Value           |
  | Momentum    | Buy winners / sell losers                 | UMD, AQR Momentum        |
  | Carry       | Buy high-yielding / sell low-yielding     | AQR Carry (multi-asset)  |
  | Defensive   | Buy low-beta quality / sell high-beta junk| BAB (Frazzini-Pedersen)  |
  | Trend       | Time-series momentum across asset classes | SG Trend Index, AQR TSMOM|
  | Vol Premium | Sell implied vol (exceeds realized)       | PUTW, VRP                |
  | Merger Arb  | Buy targets / hedge with acquirers       | MERFX spread             |

  This should be CHEAP (ARP funds charge 0.5-1.0%, not 2-and-20)

LAYER 3 — PURE ALPHA:
  Residual returns unexplained by Layers 1 and 2.
  This is the ONLY component worth paying 2-and-20 for.

  α_fund = R_fund - (Layer 1 + Layer 2)
```

### Regime-Conditional ARP Decomposition

**The key insight**: The composition of hedge fund returns (how much is alpha vs. beta)
changes dramatically across regimes.

```
GROWTH REGIME:
  L/S Equity typical decomposition:
    Traditional Beta: 60%  (equity market exposure dominates)
    Alternative Beta: 25%  (momentum, value factors contribute)
    Pure Alpha: 15%        (stock selection adds value)

CRISIS REGIME:
  L/S Equity typical decomposition:
    Traditional Beta: 85%  (beta overwhelms everything)
    Alternative Beta: 10%  (factors also crash)
    Pure Alpha: 5%         (very few managers add value in crisis)

  BUT CTA:
    Traditional Beta: -20% (short market exposure helps)
    Alternative Beta: 90%  (trend factor IS the return)
    Pure Alpha: 30%        (good CTAs capture more trend)

IMPLICATION:
  In Growth, you're paying for alpha → focus on high-alpha managers
  In Crisis, almost everything is beta → focus on WHICH betas you want
  The regime should directly influence whether you emphasize alpha or beta
```

### Implementation

```python
# Step 1: Define factor sets
trad_beta_factors = ['SPX', 'AGG', 'Credit']  # Traditional beta
arp_factors = ['Value', 'Momentum', 'Carry', 'Defensive', 'Trend', 'VRP']  # ARP

# Step 2: Run sequential regression (Fama-MacBeth style)
# First strip out traditional beta
residual_1 = R_fund - β_trad @ trad_beta_factors
# Then strip out alternative beta
residual_2 = residual_1 - β_arp @ arp_factors
# Residual_2 = pure alpha

# Step 3: Compute per regime
for regime in regimes:
    mask = regime_labels == regime
    decomp[regime] = {
        'trad_beta': β_trad[mask] @ F_trad[mask],
        'alt_beta': β_arp[mask] @ F_arp[mask],
        'alpha': residual_2[mask]
    }
```

### Allocation Implication

```
FEE-EFFICIENCY RULE:
  Traditional Beta → get from ETFs/index funds (near-zero cost)
  Alternative Beta → get from ARP products (50-100bp)
  Pure Alpha → pay hedge fund fees (200bp + 20% incentive)

REGIME-CONDITIONAL FEE BUDGET:
  Growth: allocate more fee budget to alpha-generating funds
  Crisis: reduce fee budget, increase systematic beta hedges
  Recovery: fee budget to distressed/credit alpha
```

### References
- Ardia, Barras, Gagliardini, Scaillet (2024). "Is it Alpha or Beta? Decomposing
  Hedge Fund Returns When Models are Misspecified." JFE, 154.
- Fung & Hsieh (2004). "Hedge Fund Benchmarks: A Risk-Based Approach." FAJ.

---

## 8. Regime-Switching Factor Model

**Slots into**: Application Layer (Fund-Level Analysis), replaces/extends Section 6

### The Problem with Constant-Beta Models

Billio, Getmansky, and Pelizzon (2006, 2008) proved that hedge fund factor exposures
change dramatically across regimes. A constant-beta model says "this fund has 0.3 equity
beta." In reality:

```
EMPIRICAL FINDING (Billio et al.):
  Fund XYZ (Equity L/S):
    Growth regime:  β_equity = 0.25  (well-hedged, as advertised)
    Crisis regime:  β_equity = 0.75  (hedge breaks down under stress)

  Fund ABC (Relative Value):
    Growth regime:  β_credit = 0.10  (market-neutral, as advertised)
    Crisis regime:  β_credit = 0.60  (spread positions gap against them)

CONSEQUENCE:
  A constant-beta model says the portfolio has β_equity = 0.30
  In a crisis, the TRUE beta might be 0.65 — more than double
  Your risk model is lying to you precisely when it matters most
```

### Markov-Switching Factor Model

```
R_fund_t = α_{s_t} + Σᵢ β_{i,s_t} · F_{i,t} + ε_{s_t,t}

where:
  s_t ∈ {Growth, Slowdown, Crisis, Recovery}  (from regime model)
  α_{s_t} = regime-specific alpha
  β_{i,s_t} = regime-specific factor loading
  ε_{s_t,t} ~ N(0, σ²_{s_t})  (regime-specific residual vol)
```

### Implementation

```
APPROACH A — KNOWN REGIMES (use regime labels from Jump Model/HMM):
  For each regime k:
    Run OLS: R_fund_t = α_k + Σ β_{i,k} · F_{i,t} + ε_t
    on data where regime_labels == k

  Simple, interpretable, but requires enough data in each regime

APPROACH B — JOINT ESTIMATION (if you want the factor model to discover regimes):
  Estimate Markov-switching regression jointly via EM algorithm:
    E-step: compute P(s_t=k | data, params) using forward-backward
    M-step: re-estimate α_k, β_k, σ²_k via weighted regression

  More data-efficient but harder to implement

RECOMMENDED: Approach A — use the regime labels from your Jump Model/HMM ensemble
(which are already well-calibrated with 26+ signals) and simply run conditional
regressions. This is simpler and more robust.
```

### Critical Output: Regime-Conditional Risk Aggregation

```
Portfolio β_equity in each regime:
  β_port_equity_{Growth}  = Σ w_fund · β_fund_equity_{Growth}
  β_port_equity_{Crisis}  = Σ w_fund · β_fund_equity_{Crisis}

If β_port_equity_{Crisis} >> β_port_equity_{Growth}, the portfolio
has HIDDEN CRISIS BETA that constant models miss.

RISK REPORT MUST SHOW:
  "Portfolio equity beta: 0.28 (Growth), 0.61 (Crisis)"
  Not just "Portfolio equity beta: 0.35" (misleading average)
```

### Phase-Locking Effect (Billio et al.)

During crises, even the IDIOSYNCRATIC component of hedge fund returns becomes correlated
with market stress. Billio et al. call this "phase-locking" — the regime of the residual
risk switches in tandem with market regime. This means diversification across hedge funds
breaks down more than even regime-conditional beta models suggest.

### References
- Billio, Getmansky, Pelizzon (2006). "Phase-Locking and Switching Volatility in
  Hedge Funds." Working Paper.
- Billio, Getmansky, Pelizzon (2012). "Dynamic Risk Exposures in Hedge Funds."
  Computational Statistics & Data Analysis, 56(11).

---

## 9. Drawdown Regime Overlay

**Slots into**: Risk Layer (new), between Model Layer and Allocation Engine

### Why a Drawdown Overlay Is Non-Negotiable for FoF

In practice, **drawdown is the risk metric that triggers action** in hedge fund allocating:
- Investors redeem from FoFs that breach drawdown thresholds
- FoFs redeem from underlying funds that breach drawdown thresholds
- Risk committees trigger reviews based on drawdown, not volatility
- The worst FoF failures were drawdown-driven (not vol-driven)

A macro regime model might say "Growth" while a single fund is down -15% due to
idiosyncratic issues. The drawdown overlay catches what macro models miss.

### Drawdown State Machine

```
For portfolio-level AND each fund individually:

  Track: DD_t = (W_t / max(W_s, s≤t)) - 1    (current drawdown)
  Track: DD_duration = months since last high-water mark

  DRAWDOWN STATES:

  ┌─────────┐    DD < -3%     ┌──────────┐    DD < -7%     ┌──────────┐
  │  GREEN   │ ──────────────→ │  YELLOW   │ ──────────────→ │  ORANGE  │
  │  DD < 3% │                 │  DD 3-7%  │                 │  DD 7-12%│
  │  Normal  │ ←────────────── │  Caution  │ ←────────────── │  Stress  │
  └─────────┘  Recovery >HWM  └──────────┘  DD improves    └──────────┘
                                                                │
                                                    DD < -12%   │
                                                                ▼
                                                            ┌──────────┐
                                                            │   RED    │
                                                            │  DD >12% │
                                                            │  Crisis  │
                                                            └──────────┘

  ACTIONS BY STATE:

  GREEN:   Full allocation. Regime model drives decisions.

  YELLOW:  Flag for review. Check:
           - Is drawdown idiosyncratic (single fund) or systemic (regime-driven)?
           - If idiosyncratic: fund-level review, possible reduction
           - If regime-driven: macro regime model takes precedence

  ORANGE:  Mandatory risk reduction.
           - Reduce gross exposure by 25%
           - Cut worst-performing positions first
           - Override regime model if it still says "Growth"
           - Duration threshold: if DD_duration > 6 months at Orange → escalate to Red

  RED:     Capital preservation mode.
           - Target minimum risk allocation
           - Exit illiquid positions (begin redemption process)
           - Hold only: CTA, Macro, cash, tail hedges
           - This is a HARD override of the regime model
```

### CPPI-Style Position Sizing Under Drawdown Constraint

```
Optimal allocation under drawdown constraint (Grossman-Zhou):

  w_risky_t = m · (W_t - α · M_t) / W_t

  where:
    W_t = current portfolio value
    M_t = high-water mark (running maximum)
    α = drawdown floor (e.g., 0.88 for max 12% drawdown)
    m = multiplier (calibrated, typically 3-5)

  (W_t - α·M_t) = "cushion" — distance to drawdown limit

  Properties:
    - As drawdown deepens → cushion shrinks → allocation to risky strategies drops
    - At drawdown limit → w_risky = 0 (full cash)
    - Automatically recovers as portfolio recovers
    - No parameter tuning needed beyond α and m

  REGIME INTERACTION:
    Multiply by regime confidence:
    w_final = w_CPPI · regime_confidence_scalar

    Growth + low drawdown → maximum risk
    Crisis + deep drawdown → minimum risk (double de-risk)
```

### Fund-Level Drawdown Monitoring

```
For each fund in the portfolio:

  1. Track DD_fund_t continuously (via nowcasting if monthly-reporting)
  2. Alert thresholds:
     - DD_fund > -5%:  yellow flag
     - DD_fund > -10%: orange flag → begin exit analysis
     - DD_fund > -15%: red flag → submit redemption
  3. DD RELATIVE TO STRATEGY:
     - Compare fund DD to strategy index DD
     - If fund DD >> strategy DD → manager-specific problem → redeem
     - If fund DD ≈ strategy DD → market-driven → regime model decides
  4. DD DURATION:
     - Track how long since HWM
     - Duration > 12 months → structural problem, not cyclical
```

### References
- Grossman & Zhou (1993). "Optimal Investment Strategies for Controlling Drawdowns."
  Mathematical Finance, 3(3).

---

## 10. Functional Fund Classification (JP Morgan / GIC Framework)

**Slots into**: Strategy-Regime Mapping (extends Section 5.1)

### Why Traditional Strategy Labels Are Insufficient

The original design classifies funds by strategy (L/S Equity, Macro, CTA, etc.). But
JP Morgan and GIC's research shows that functional classification — **what role the fund
plays in the portfolio** — is more useful for regime-aware allocation.

### The Four Functional Buckets

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUNCTIONAL CLASSIFICATION                     │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  LOSS MITIGATION     │  │  EQUITY DIVERSIFIER  │            │
│  │                      │  │                      │            │
│  │  Purpose: Protect    │  │  Purpose: Diversify  │            │
│  │  in drawdowns        │  │  equity risk         │            │
│  │                      │  │                      │            │
│  │  Strategies:         │  │  Strategies:         │            │
│  │  • CTA/Trend         │  │  • Global Macro      │            │
│  │  • Long Volatility   │  │  • FI Relative Value │            │
│  │  • Tail Risk Hedging │  │  • Market Neutral    │            │
│  │  • Macro (defensive) │  │  • Multi-Strategy    │            │
│  │                      │  │                      │            │
│  │  Expected Crisis     │  │  Expected Crisis     │            │
│  │  return: POSITIVE    │  │  return: FLAT/SMALL  │            │
│  │                      │  │  NEGATIVE            │            │
│  │  Correlation to      │  │  Correlation to      │            │
│  │  equity: NEGATIVE    │  │  equity: LOW         │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  EQUITY COMPLEMENT   │  │  EQUITY SUBSTITUTE   │            │
│  │                      │  │                      │            │
│  │  Purpose: Enhance    │  │  Purpose: Replace    │            │
│  │  equity returns with │  │  equity allocation   │            │
│  │  better risk-adjust  │  │  with lower beta     │            │
│  │                      │  │                      │            │
│  │  Strategies:         │  │  Strategies:         │            │
│  │  • L/S Equity (low   │  │  • L/S Equity (high  │            │
│  │    net exposure)     │  │    net exposure)     │            │
│  │  • Equity Event      │  │  • Activist          │            │
│  │    Driven            │  │  • Long-biased       │            │
│  │  • Statistical Arb   │  │  • Sector L/S        │            │
│  │                      │  │                      │            │
│  │  Expected Crisis     │  │  Expected Crisis     │            │
│  │  return: NEGATIVE    │  │  return: VERY        │            │
│  │  (but < equity)     │  │  NEGATIVE            │            │
│  │                      │  │                      │            │
│  │  Correlation to      │  │  Correlation to      │            │
│  │  equity: MODERATE    │  │  equity: HIGH        │            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Regime-Conditional Allocation Ranges by Functional Bucket

```
| Bucket              | Growth  | Slowdown | Crisis  | Recovery |
|---------------------|---------|----------|---------|----------|
| Loss Mitigation     | 10-15%  | 15-25%  | 25-40%  | 10-15%  |
| Equity Diversifier  | 20-30%  | 25-35%  | 25-35%  | 20-30%  |
| Equity Complement   | 30-40%  | 20-30%  | 10-20%  | 25-35%  |
| Equity Substitute   | 20-30%  | 10-20%  | 5-10%   | 20-30%  |
                        ════════  ════════  ════════  ════════
| Total               | 100%    | 100%    | 100%    | 100%    |

KEY SHIFT:
  Growth → Crisis: Loss Mitigation goes from 10% to 40%
                    Equity Substitute goes from 25% to 5%
                    This is the core regime-driven rotation
```

### How to Classify Your Funds

```
For each fund, compute:
  1. β_equity (equity beta from regime-switching model)
  2. Crisis_return = average return in Crisis regime months
  3. Equity_correlation = correlation with S&P 500

Classification rules:
  β_equity < -0.1 AND Crisis_return > 0    → Loss Mitigation
  β_equity < 0.3  AND |Equity_corr| < 0.3  → Equity Diversifier
  β_equity 0.3-0.5 AND Equity_corr 0.3-0.6 → Equity Complement
  β_equity > 0.5 AND Equity_corr > 0.6     → Equity Substitute
```

### Reference
- JP Morgan Asset Management & GIC (2024). "Building a Hedge Fund Allocation:
  Integrating Top-down and Bottom-up Perspectives."

---

## 11. Institutional Validation — What the Best Actually Use

### Two Sigma: Gaussian Mixture Model on 17-Factor Lens

Two Sigma published their regime framework using a **Gaussian Mixture Model (GMM)** on
their proprietary 17-factor lens, identifying four regimes:

| Regime | Characteristics |
|--------|----------------|
| **Steady State** | Moderate vol, positive returns, normal correlations |
| **Crisis** | High vol, negative equity, correlation spike |
| **Inflation** | Rising rates, commodities outperform, duration negative |
| **Walking on Ice** | Low vol but fragile, low returns, elevated tail risk |

The "Walking on Ice" regime is particularly insightful — it's what our system captures
when Crowding is High + Macro is Growth. Looks calm but is maximally fragile.

### State Street Global Advisors (SSGA): Market Regime Indicator

SSGA's **MRI (Market Regime Indicator)** uses:
- Equity implied volatility
- Currency implied volatility
- Credit spreads
- Economic growth indicators
- Fed balance sheet
- HMM for tactical allocation

Their 2025 paper confirms: ML-identified regimes explain 30-year asset performance better
than traditional approaches. They use **GARCH + HMM** combination.

### Common Threads Across Top Allocators

```
EVERY top institutional allocator:

  1. ✓ Separates alpha from beta systematically (ARP decomposition)
  2. ✓ Uses some form of regime model (HMM, GMM, or Jump Model)
  3. ✓ Has real-time monitoring beyond monthly NAVs (nowcasting)
  4. ✓ Manages drawdowns systematically (not just macro regime)
  5. ✓ Uses functional classification (not just strategy labels)
  6. ✓ Monitors connectedness / contagion across managers
  7. ✓ Accounts for return smoothing in risk measurement
  8. ✓ Uses forward-looking signals (implied vol/correlation, not just realized)
```

---

## 12. Updated Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                            DATA LAYER                                       ║
║                                                                              ║
║  Bloomberg API │ Fund Returns │ HF Indices │ Macro │ Factor Data │ Options  ║
║                                    │                                         ║
║                        ┌───────────┴───────────┐                            ║
║                        │   RETURN UNSMOOTHING   │  ← NEW                    ║
║                        │   (GLM MA(2) inversion)│                            ║
║                        └───────────┬───────────┘                            ║
║                                    │                                         ║
║                        ┌───────────┴───────────┐                            ║
║                        │   DAILY NOWCASTING     │  ← NEW                    ║
║                        │   (Kalman Filter DSA)  │                            ║
║                        └───────────┬───────────┘                            ║
╚════════════════════════════════════╬═════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FEATURE ENGINEERING (32 Signals)                         ║
║                                                                              ║
║  Original 26 Signals (unchanged)                                             ║
║  + Signal 27: Factor Crowding (CoMetric)                          ← NEW     ║
║  + Signal 28: Implied Correlation + CRP Gap                       ← NEW     ║
║  + Signal 29: DY Total Connectedness Index                        ← NEW     ║
║  + Signal 30: Contagion Net Directionality (transmitter/receiver) ← NEW     ║
║  + Signal 31: Smoothing-Adjusted Strategy Correlations            ← NEW     ║
║  + Signal 32: Copper/Gold Ratio (growth vs safety appetite)       ← NEW     ║
╚════════════════════════════════════╬═════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MODEL LAYER (Ensemble)                               ║
║                                                                              ║
║  Tier 1: Statistical Jump Model (PRIMARY)           — unchanged              ║
║  Tier 2: Bayesian HMM Student-t (SECONDARY)         — unchanged              ║
║  Tier 3: BOCPD (ALERT LAYER)                        — upgraded to Score-     ║
║                                                        Driven BOCPD          ║
║  Ensemble: Confidence-Weighted Consensus             — unchanged              ║
║                                                                              ║
║  + CROWDING REGIME OVERLAY (separate dimension)       ← NEW                  ║
║    Crowded / Moderate / Uncrowded                                            ║
║    Interacts with macro regime (Crowded+Growth = "Walking on Ice")           ║
╚════════════════════════════════════╬═════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RISK OVERLAY LAYER                    ← NEW          ║
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │  DRAWDOWN REGIME OVERLAY                                       │         ║
║  │  Green / Yellow / Orange / Red                                 │         ║
║  │  CPPI-style position sizing: w = m · (cushion / wealth)        │         ║
║  │  HARD OVERRIDE: Red drawdown → minimum risk regardless of      │         ║
║  │  macro regime                                                  │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
║  ┌────────────────────────────────────────────────────────────────┐         ║
║  │  CONTAGION EXIT ORDERING                                       │         ║
║  │  In crisis: exit RV/Credit first → ED → L/S Eq → hold Macro/CTA│        ║
║  │  Based on DY Net Connectedness + prime broker concentration    │         ║
║  └────────────────────────────────────────────────────────────────┘         ║
╚════════════════════════════════════╬═════════════════════════════════════════╝
                                     ║
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        APPLICATION LAYER                                     ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────┐            ║
║  │  ARP DECOMPOSITION (per fund, per regime)                    │  ← NEW    ║
║  │  Traditional Beta | Alternative Beta | Pure Alpha            │           ║
║  │  Fee-efficiency analysis: are you paying 2&20 for beta?      │           ║
║  └─────────────────────────────────────────────────────────────┘            ║
║  ┌─────────────────────────────────────────────────────────────┐            ║
║  │  REGIME-SWITCHING FACTOR MODEL (per fund)                    │  ← NEW    ║
║  │  β_{i,regime} — factor loadings per regime state             │           ║
║  │  "True" crisis beta vs. reported beta                        │           ║
║  └─────────────────────────────────────────────────────────────┘            ║
║  ┌─────────────────────────────────────────────────────────────┐            ║
║  │  FUNCTIONAL CLASSIFICATION                                   │  ← NEW    ║
║  │  Loss Mitigation | Eq Diversifier | Eq Complement | Eq Sub  │           ║
║  │  Regime-conditional allocation ranges per bucket             │           ║
║  └─────────────────────────────────────────────────────────────┘            ║
║                                                                              ║
║  Strategy-Regime Mapping     — enhanced with unsmoothed data                 ║
║  Fund-Level Analysis         — enhanced with regime-switching betas          ║
║  Allocation Engine           — enhanced with drawdown overlay + contagion    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 13. Updated Feature Matrix (32 Signals)

Original 26 signals (unchanged) PLUS:

| # | Signal | Source | Category | Primary Use |
|---|--------|--------|----------|-------------|
| 27 | Factor Crowding (CoMetric) | Fund residual correlations | Crowding | Detect crowded positions before unwind |
| 28a | Implied Correlation %ile | CBOE ICJ/KCJ | Forward-looking | Leading correlation regime signal |
| 28b | Correlation Risk Premium gap | Implied - Realized | Forward-looking | Stress building before it materializes |
| 29 | DY Total Connectedness | VAR on strategy returns | Contagion | System-wide stress interconnection |
| 30 | Net Connectedness vector | DY directional | Contagion | Who is transmitting stress |
| 31 | Smoothing-adjusted corr | Unsmoothed returns | Data quality | True (not reported) correlation |
| 32 | Copper/Gold ratio | HG1/GC1 | Cross-asset | Real-time growth vs. safety appetite |

**Total: 32 signals (26 original + 6 new)**

---

## 14. Updated Build Phases

### Phase 0: Data Foundation (Week 1) — NEW
```
Deliverables:
  ✦ Return unsmoothing module (GLM MA(2) estimation + inversion)
  ✦ Apply to all fund returns and HF index returns
  ✦ Compute smoothing index ξ for each fund (flag ξ < 0.7 as heavily smoothed)
  ✦ Compare: raw vs unsmoothed vol, Sharpe, correlation for each fund
  ✦ Nowcasting module: Kalman Filter DSA per fund
  ✦ Validate nowcast accuracy on 1 year of held-out data

Notebook: 00_data_foundation.ipynb
```

### Phase 1: Feature Engineering (Weeks 2-3)
```
Same as original + add signals 27-32:
  ✦ Factor crowding (CoMetric) computation
  ✦ Implied correlation data pipeline + CRP gap
  ✦ Diebold-Yilmaz connectedness (rolling VAR + GFEVD)
  ✦ Validate all 32 signals against known crisis periods

Notebooks: 01_data_exploration.ipynb, 02_feature_signals.ipynb
```

### Phase 2: Regime Model (Weeks 3-5) — same as original
```
  + Add crowding overlay as separate regime dimension
  + Upgrade BOCPD to score-driven variant
```

### Phase 3: Fund Analysis (Weeks 5-8) — EXPANDED
```
Original deliverables +
  ✦ ARP decomposition per fund (3-layer: trad beta / alt beta / alpha)
  ✦ Regime-switching factor model (β per regime per fund)
  ✦ Functional classification (Loss Mitigation / Eq Div / Eq Comp / Eq Sub)
  ✦ Regime-conditional ARP decomposition (how alpha/beta split changes)
  ✦ Fee-efficiency analysis: which funds are paid for alpha but deliver beta?
  ✦ Contagion analysis: DY connectedness + prime broker concentration

Notebooks: 04_strategy_mapping.ipynb, 05_fund_analysis.ipynb,
           05b_arp_decomposition.ipynb  (NEW)
```

### Phase 4: Allocation + Risk (Weeks 8-11) — EXPANDED
```
Original deliverables +
  ✦ Drawdown overlay implementation (state machine + CPPI sizing)
  ✦ Contagion-aware exit ordering
  ✦ Functional bucket allocation ranges
  ✦ Integrated backtest: regime model + drawdown overlay + contagion ordering
  ✦ Stress test: simulate Aug 2007 (crowding), 2008 (systemic), 2020 (speed)

Notebooks: 06_allocation_backtest.ipynb, 06b_drawdown_overlay.ipynb (NEW)
```

### Phase 5: Production (Weeks 11-13)
```
Same as original + nowcasting pipeline for daily monitoring
```

---

## 15. Complete Reference Library

### Foundational Papers

| Paper | Authors | Year | Component |
|-------|---------|------|-----------|
| An Econometric Model of Serial Correlation and Illiquidity in Hedge Fund Returns | Getmansky, Lo, Makarov | 2004 | Return Unsmoothing |
| Principal Components as a Measure of Systemic Risk | Kritzman, Li, Page, Rigobon | 2010 | Absorption Ratio |
| Managing Diversification | Meucci | 2009 | ENB |
| Correlation Surprise | Kinlaw, Turkington | 2012 | Turbulence Decomposition |
| Bayesian Online Changepoint Detection | Adams, MacKay | 2007 | BOCPD |
| Learning HMMs with Persistent States by Penalizing Jumps | Nystrup et al. | 2020 | Statistical Jump Model |
| Downside Risk Reduction: A Statistical Jump Model Approach | Shu, Mulvey | 2024 | Jump Model for Allocation |
| Hedge Fund Replication Using Strategy-Specific Factors | Fung, Hsieh | 2004 | 7-Factor Model |

### Hedge Fund Specific

| Paper | Authors | Year | Component |
|-------|---------|------|-----------|
| What Happened to the Quants in August 2007? | Khandani, Lo | 2007 | Factor Crowding |
| The Impact of Crowding in Alternative Risk Premia Investing | Baltas | 2019 | CoMetric |
| Is it Alpha or Beta? Decomposing Hedge Fund Returns | Ardia, Barras, Gagliardini, Scaillet | 2024 | ARP Decomposition |
| Dynamic Risk Exposures in Hedge Funds | Billio, Getmansky, Pelizzon | 2012 | Regime-Switching Betas |
| Econometric Measures of Connectedness and Systemic Risk | Billio, Getmansky, Lo, Pelizzon | 2012 | Contagion |
| Hedge Fund Contagion and Liquidity Shocks | Boyson, Stahel, Stulz | 2010 | Contagion Hierarchy |
| Monitoring Daily Hedge Fund Performance | Wermers | 2014 | Nowcasting |
| Optimal Investment Strategies for Controlling Drawdowns | Grossman, Zhou | 1993 | Drawdown Overlay |

### Institutional Frameworks

| Paper | Authors | Year | Component |
|-------|---------|------|-----------|
| Building a Hedge Fund Allocation: Integrating Top-down and Bottom-up | JP Morgan AM, GIC | 2024 | Functional Classification |
| A Machine Learning Approach to Regime Modeling | Two Sigma | 2023 | GMM + Factor Lens |
| Decoding Market Regimes with Machine Learning | SSGA | 2025 | MRI + HMM |
| The Price of Correlation Risk | Driessen, Maenhout, Vilkov | 2009 | Implied Correlation |
| On the Network Topology of Variance Decompositions | Diebold, Yilmaz | 2014 | Connectedness Framework |

### 2024-2025 Cutting Edge

| Paper | Authors | Year | Component |
|-------|---------|------|-----------|
| RHINE: Regime-Switching Model with Nonlinear Representation | SIAM | 2024 | Kernel-based regime detection |
| Bayesian Autoregressive Online Change-Point Detection with Time-Varying Parameters | arXiv | 2024 | Score-Driven BOCPD |
| Dynamic Factor Allocation Leveraging Regime-Switching Signals | arXiv | 2024 | Factor timing with JM/HMM |
| Representation Learning for Regime Detection in Block Hierarchical Markets | arXiv | 2024 | SPDNet on correlation manifolds |

---

## Summary: What Changed from V1 to V2

```
V1 (Original DESIGN.md):
  ✓ 26 signals across 6 categories
  ✓ Jump Model + HMM + BOCPD ensemble
  ✓ Strategy-regime mapping
  ✓ Fund-level regime analysis
  ✓ BL + Risk Parity allocation
  ✓ Backtesting framework

  ✗ Treated HF returns as clean (they're smoothed)
  ✗ No intra-month visibility (nowcasting)
  ✗ Blind to crowding regime
  ✗ No forward-looking correlation signal
  ✗ No contagion / exit ordering
  ✗ No alpha/beta separation
  ✗ Constant-beta assumption (wrong in crisis)
  ✗ No drawdown management layer
  ✗ Traditional strategy labels only

V2 (This Upgrade):
  ✓ Everything in V1
  ✓ Return unsmoothing before any analysis
  ✓ Daily nowcasting via Kalman Filter DSA
  ✓ Factor crowding detection (CoMetric)
  ✓ Implied correlation + CRP gap (forward-looking)
  ✓ Diebold-Yilmaz contagion framework + exit ordering
  ✓ 3-layer ARP decomposition (trad beta / alt beta / alpha)
  ✓ Regime-switching factor model (regime-conditional betas)
  ✓ Drawdown regime overlay with CPPI sizing
  ✓ Functional fund classification (JP Morgan/GIC framework)
  ✓ 32 signals (6 new)
  ✓ Validated against Two Sigma, SSGA, JP Morgan/GIC, Man FRM approaches

This is the institutional-grade system.
```
