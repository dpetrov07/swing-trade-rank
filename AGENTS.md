# AGENTS.md — Swing Trade Rank

## Project purpose

Swing Trade Rank is a quantitative equity-research project for testing whether simple and machine-learning signals can rank liquid U.S. stocks by future cross-sectional returns.

The immediate goal is NOT to build a trading platform, broker integration, dashboard, options system, or autonomous trading bot.

The immediate goal is to answer one research question correctly:

> Can sector-relative momentum rank liquid U.S. stocks by their future 10-trading-day sector-relative returns?

Only after the baseline is implemented and validated should the project test whether machine learning adds incremental predictive value.

---

## Core principles

1. Research validity is more important than feature count.
2. Prevent look-ahead leakage at every stage.
3. Use chronological validation only.
4. Prefer simple, inspectable baselines before ML.
5. Every experiment must be reproducible from configuration.
6. Failed experiments must be saved, not silently discarded.
7. No result should be described as profitable or predictive until it passes out-of-sample tests.
8. Keep V1 small. Do not add infrastructure that is not necessary for the current experiment.
9. Never silently change a hypothesis, target, universe, horizon, or evaluation rule after seeing results. Create a new experiment instead.
10. The code should make timing assumptions explicit.

---

## Project name

Project: `Swing Trade Rank`

Suggested repository name:

```text
swing-trade-rank
```

---

## V1 research questions

### EXP-001 — Relative momentum baseline

Primary hypothesis:

> Stocks with stronger recent momentum relative to their sector tend to outperform their sector over the following 10 trading days.

Primary target:

```text
future_sector_relative_return_10d
=
future_stock_total_return_10d
-
future_sector_return_10d
```

Primary prediction frequency:

```text
weekly
```

Use one consistent weekly signal date.

Signals are formed only from information available as of the signal timestamp.

If a signal uses the closing price on date `t`, the simulated strategy may NOT transact at that same closing price.

Default execution assumption:

```text
signal: close of t
execution: next tradable session open
```

If the data source cannot support next-open execution, document the alternative explicitly and do not mix signal and execution prices.

### EXP-002 — Linear baseline

Use the same:
- universe
- prediction dates
- features
- target
- folds
- execution assumptions

Compare a simple linear/ridge model to EXP-001.

### EXP-003 — Gradient-boosted model

Use the same experimental setup and evaluate whether XGBoost adds incremental cross-sectional ranking power.

Start with `XGBRegressor`.

Do not introduce `XGBRanker` until the regression baseline is working and correctly evaluated. A ranking objective may be investigated later as a separate experiment.

---

## V1 universe

Target roughly 300–1,000 liquid U.S. common stocks.

Initial eligibility rules:

```text
security type: U.S. common stock
price: > $5
trailing median daily dollar volume: > $10M
```

Market-cap filtering is optional in the first implementation because reliable historical point-in-time market-cap data may require a higher-quality data source.

If point-in-time market cap is available, use:

```text
market cap > $1B
```

### Critical universe rule

Do NOT define historical membership from today's S&P 500, today's Russell membership, or today's ticker list and assume those securities existed throughout history.

The code must make the data limitation explicit if active and delisted historical securities are not available.

A prototype using only currently available tickers is allowed for pipeline development, but all reports must label it:

```text
SURVIVORSHIP-BIASED RESEARCH PROTOTYPE
```

until a point-in-time universe is used.

---

## Data requirements

Keep raw source data immutable.

Prefer a source that provides:
- historical OHLCV
- adjusted or total-return price history
- unadjusted prices
- delisted securities
- ticker changes
- splits/dividends/corporate actions
- security type
- sector classification
- historical metadata where possible

For serious historical validation, a dataset with active and delisted securities is preferred.

Do not pretend a free/current-ticker-only dataset is survivorship-safe.

### Price conventions

Use adjusted/total-return prices for return calculations when appropriate.

Use unadjusted or vendor-recommended fields for:
- actual tradable prices
- execution assumptions
- price eligibility filters
- dollar-volume calculations

Do not combine adjusted price with raw volume without understanding the vendor's adjustment convention.

Document the convention in `docs/data.md`.

---

## Proposed repository structure

Start smaller than a production system.

```text
swing-trade-rank/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── config/
│   ├── universe.yaml
│   └── experiments/
│       ├── exp_001_momentum.yaml
│       ├── exp_002_ridge.yaml
│       └── exp_003_xgboost.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── src/
│   └── swing_trade_rank/
│       ├── data/
│       │   ├── prices.py
│       │   ├── universe.py
│       │   └── sectors.py
│       │
│       ├── features/
│       │   ├── momentum.py
│       │   ├── volatility.py
│       │   └── volume.py
│       │
│       ├── targets.py
│       ├── preprocessing.py
│       │
│       ├── models/
│       │   ├── baseline.py
│       │   ├── linear.py
│       │   └── xgboost_model.py
│       │
│       ├── evaluation/
│       │   ├── ic.py
│       │   ├── deciles.py
│       │   ├── signal_decay.py
│       │   └── walk_forward.py
│       │
│       ├── portfolio/
│       │   ├── simulator.py
│       │   ├── costs.py
│       │   └── metrics.py
│       │
│       └── experiments/
│           └── runner.py
│
├── scripts/
│   ├── build_dataset.py
│   └── run_experiment.py
│
├── reports/
│   └── experiments/
│
├── tests/
│
└── docs/
    ├── data.md
    └── methodology.md
```

Do not create empty placeholder modules unless they are needed by the current milestone.

---

## Initial feature set

Keep V1 near 10 features.

Candidate features:

```text
mom_5d
mom_20d
mom_60d
mom_120d

mom_20d_vs_sector
mom_60d_vs_sector

mom_20d_vs_market

distance_252d_high

volatility_20d

volume_ratio_20d

momentum_acceleration_20d
```

Where:

```text
momentum_acceleration_20d
=
return(t-20, t)
-
return(t-40, t-20)
```

Not every candidate must survive into the final feature set.

Avoid adding RSI, MACD, Bollinger Bands, dozens of TA indicators, or alternative data merely to increase model complexity.

---

## Cross-sectional preprocessing

At each prediction date, preprocessing must be performed using only that date's cross section.

For ML experiments:

1. calculate raw features using only historical data
2. optionally winsorize extreme feature values cross-sectionally
3. standardize or rank-normalize features cross-sectionally
4. fit any learned preprocessing using training data only where applicable

Recommended first approach:

```text
cross-sectional percentile rank
```

or robust z-score for continuous features.

Never normalize using statistics calculated across future dates.

Missing-value policy must be explicit.

Do not silently fill missing values with future information.

---

## Sector handling

Sector-relative features and targets must use sector information available for the relevant security/date as accurately as the dataset permits.

For the first baseline:

```text
sector_relative_momentum
=
stock_return
-
sector_return
```

Sector return can initially be represented by a mapped liquid sector ETF if historical constituent-level sector aggregation is too complex.

Document this choice.

Do not claim ETF-relative and constituent-sector-relative calculations are identical.

---

## Targets

Primary target is fixed before analysis:

```text
10 trading-day sector-relative forward return
```

Also calculate diagnostic horizons:

```text
1d
5d
20d
40d
```

These are diagnostics, not permission to select whichever horizon looks best on the full dataset.

If another horizon appears stronger, define a NEW experiment and validate that choice on a later untouched period.

### Target timing

If the strategy executes at next-session open, the portfolio backtest must measure realizable returns from the assumed execution timestamp.

Research labels and portfolio returns may therefore use slightly different definitions, but the distinction must be explicit.

---

## EXP-001 baseline

Start with one signal:

```text
signal = 60d sector-relative momentum
```

At each prediction date:

1. determine eligible universe
2. calculate signal
3. rank securities cross-sectionally
4. divide securities into deciles
5. calculate future target return by decile
6. calculate Rank IC
7. save results and plots

Primary EXP-001 outputs:

```text
mean Rank IC
median Rank IC
IC standard deviation
IC information ratio
percent of periods IC > 0

mean target return by decile
D10 - D1 spread
D10 - universe-average spread

IC by year
IC by sector
signal decay
sample counts
```

Use Spearman correlation for Rank IC unless an experiment explicitly specifies otherwise.

---

## Sanity checks required before ML

EXP-001 is not complete until tests confirm:

- no future prices enter any feature
- future-return labels are shifted in the correct direction
- prediction dates contain the expected number of securities
- ranking is performed independently within each prediction date
- decile labels increase in the intended direction
- sector-relative target arithmetic is correct
- missing dates/tickers do not silently misalign
- duplicate date/ticker rows are rejected
- train/test boundaries do not overlap through forward labels

Also run at least one intentionally meaningless baseline:

```text
random score with fixed seed
```

Expected long-run Rank IC should be approximately zero.

If the random baseline looks consistently predictive, investigate the pipeline before proceeding.

---

## ML model order

Do not jump directly to XGBoost.

Use:

```text
EXP-001: raw momentum ranking
EXP-002: Ridge / linear model
EXP-003: XGBoost
```

The important question is incremental value.

Compare all models on identical out-of-sample observations.

Primary ML evaluation metric:

```text
out-of-sample Spearman Rank IC
```

Secondary:
- ICIR
- decile monotonicity
- top-minus-bottom spread
- top-decile excess return
- stability by year
- stability by regime/sector
- turnover once portfolio simulation exists

Do not optimize for in-sample R².

---

## Walk-forward validation

Never use random `train_test_split()` for model evaluation.

Use expanding-window or rolling-window validation.

Example:

```text
train: 2014–2018
test:  2019

train: 2014–2019
test:  2020

train: 2014–2020
test:  2021
```

The exact dates depend on available history.

### Embargo / purging

Because the primary label looks 10 trading days forward, prevent training observations near a test boundary from using returns that overlap the test period.

Use at least a 10-trading-day label embargo.

Implement this as reusable code and test it.

---

## Hyperparameter policy

Keep tuning intentionally limited.

For the first XGBoost experiment, use a small predefined search space.

Do not run hundreds of configurations.

Do not repeatedly inspect test-period results while tuning.

Preferred process:

```text
training window
    ↓
inner time-series validation
    ↓
select parameters
    ↓
evaluate once on outer test window
```

Record every attempted configuration.

---

## Experiment registry

Every run must create a unique output directory, for example:

```text
reports/experiments/EXP-003_2026-08-09T071500Z/
```

Save:

```text
config.yaml
metrics.json
predictions.parquet
fold_metrics.csv
decile_returns.csv
ic_series.csv
feature_importance.csv   # when applicable
equity_curve.csv         # once portfolio exists
plots/
run_metadata.json
```

`run_metadata.json` should contain:
- experiment ID
- git commit hash if available
- timestamp
- Python version
- package versions
- random seed
- input dataset identifier/hash
- train/test date ranges

Never overwrite a prior experiment directory.

---

## Portfolio simulation

Do not build this until ranking quality has been validated.

Initial portfolio:

```text
long only
weekly rebalance
equal weight
top 20 stocks OR top decile, whichever is smaller and sensible for the universe
```

Prefer top 20 over top 10 initially because it reduces idiosyncratic noise.

Add a configurable maximum sector weight, default:

```text
25%
```

Add transaction costs.

Initial sensitivity:

```text
0 bps
5 bps
10 bps
20 bps
50 bps
```

Costs should be applied to traded notional, not total portfolio value regardless of turnover.

Later, add turnover-reducing entry/exit buffers, but not in the first simulator.

---

## Portfolio benchmarks

Compare against:

```text
SPY
equal-weight eligible universe
simple 60d sector-relative momentum portfolio
ML portfolio
```

A random portfolio may be included as a sanity check, but is not a meaningful investment benchmark.

Report:

```text
CAGR
annualized volatility
Sharpe
Sortino
max drawdown
turnover
beta
alpha
hit rate
average number of holdings
sector exposure
```

Do not interpret alpha from a weak or mismatched regression as proof of market-neutral alpha.

---

## V1 completion criteria

The MVP is complete when this works from a clean environment:

```bash
python scripts/run_experiment.py \
  --config config/experiments/exp_003_xgboost.yaml
```

and produces a reproducible out-of-sample report comparing:

```text
relative-momentum baseline
vs
ridge/linear model
vs
XGBoost
```

with:

```text
Rank IC
IC by year
decile returns
signal decay
walk-forward fold results
portfolio results
cost sensitivity
```

Required plots:

```text
decile return profile
IC through time
IC by year
signal decay
equity curve
drawdown
cost sensitivity
feature importance for ML
```

The MVP is NOT judged by whether the strategy beats SPY.

A valid negative result is still a successful project if the experiment is correct and informative.

---

## Deferred work

Do not implement any of these during initial V1:

```text
React
FastAPI
PostgreSQL
broker integration
live trading
options
LLM stock picks
LLM news analysis
earnings NLP
alternative data
deep learning
reinforcement learning
real-time streaming
multi-agent architecture
portfolio optimization
```

They are roadmap items, not current tasks.

---

## Roadmap after V1

### V1.1 — Earnings continuation / PEAD

Test:
- earnings surprise
- revenue surprise
- earnings gap
- abnormal volume
- pre-earnings momentum
- guidance data if reliably available

Compare:

```text
momentum
earnings
momentum + earnings
```

### V1.2 — LLM signal experiment

Only after the structured-data pipeline is stable.

Two separate experiments are useful:

1. direct LLM picks under the same universe/horizon constraints
2. LLM-extracted structured features from filings/news/earnings

Store every LLM prediction permanently with:
- timestamp
- prompt
- model
- input sources
- output
- eligible universe

Never retrospectively regenerate historical LLM predictions using information that was not available at the prediction timestamp.

### V2 — Multi-strategy research

Potential strategies:

```text
momentum
earnings continuation
short-term reversal
quality/fundamental
LLM-derived qualitative signal
```

Combine only signals that show standalone out-of-sample evidence.

### V3 — Paper trading

Only after:
- walk-forward validation
- cost analysis
- robustness analysis
- reproducible research pipeline

Then add:
- broker paper account
- scheduler
- order generation
- execution logging
- monitoring

### V4 — Tiny live test

Live capital is for testing execution and system behavior, not proving the historical model.

Use very small capital only after a meaningful paper/forward-test period.

### V5 — Options research

Options are a separate research problem.

Do not assume a good stock-ranking signal automatically implies profitable calls.

Compare stock vs options expressions only after the underlying equity signal is credible.

---

## Coding standards for Codex

### General

- Python 3.12+ unless dependency compatibility requires otherwise.
- Use type hints for public functions.
- Prefer small pure functions for feature/target calculations.
- Use `pathlib.Path`.
- Use `logging`, not scattered `print()` statements inside library code.
- Keep configuration outside model code.
- Prefer Parquet for large tabular intermediate datasets.
- Avoid unnecessary classes.
- Avoid premature abstractions.
- Avoid hidden global state.
- Seed randomized operations.

### DataFrames

Use either Pandas or Polars consistently.

Default to Pandas for V1 unless performance becomes a real bottleneck.

Do not rewrite the project in Polars merely for perceived speed.

Canonical panel keys:

```text
date
ticker
```

Assert uniqueness whenever a table is expected to be one row per date/ticker.

Sort by:

```text
ticker, date
```

before time-series operations.

### Testing

Every important financial transformation needs a small deterministic unit test.

Especially test:
- lagging
- rolling windows
- forward returns
- sector-relative returns
- ranking
- decile assignment
- embargo logic
- transaction costs
- turnover
- execution timing

Prefer tiny hand-checkable synthetic datasets.

### Numerical correctness

Do not suppress warnings just to make tests pass.

Do not silently coerce infinities to values without documenting why.

Check denominators.

Handle insufficient lookback explicitly.

---

## Codex workflow

When implementing a task:

1. Read this file.
2. Inspect existing code before creating new files.
3. Implement the smallest coherent change.
4. Add/update tests.
5. Run the relevant tests.
6. Report what changed.
7. Report assumptions or unresolved data-quality concerns.
8. Do NOT expand scope unless explicitly requested.

If a requested task conflicts with research validity, point out the issue before implementing a misleading shortcut.

---

## First implementation milestone

Do only this first:

1. initialize project/package structure
2. add configuration loading
3. ingest a sample historical OHLCV dataset
4. validate `date × ticker` uniqueness
5. calculate:
   - 20d return
   - 60d return
   - 120d return
   - 20d volatility
   - 20d volume ratio
6. map sectors
7. calculate sector-relative 60d momentum
8. calculate forward sector-relative returns:
   - 1d
   - 5d
   - 10d
   - 20d
   - 40d
9. sample one consistent prediction date per week
10. rank 60d sector-relative momentum into deciles
11. calculate mean 10d sector-relative return by decile
12. calculate weekly Spearman Rank IC
13. save:
   - decile results CSV
   - IC series CSV
   - decile plot
   - IC summary
14. add unit tests for timing and alignment

Then STOP.

Do not implement Ridge, XGBoost, portfolio simulation, broker code, LLMs, or UI until EXP-001 results have been inspected.

---

## Definition of success for the first milestone

Success is NOT:

```text
high returns
```

Success is:

```text
a correct, leakage-free, reproducible answer to:
"Does 60-day sector-relative momentum rank future 10-day sector-relative returns in this dataset?"
```

If the answer is no, save the result.

That is valid research.
