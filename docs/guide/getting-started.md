# Getting Started

## Installation

```bash
pip install rols
```

Requires Python 3.10+ and numpy / pandas.

For development:

```bash
git clone https://github.com/GabinTB/rOLS.git
cd rOLS
uv sync --all-groups
uv run pytest
```

---

## Quick start

```python
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web
from rols import RollingOLS

# Loading some factors
factor_df = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start=start).pct_change().dropna()
factors = factor_df.columns.tolist()

# Loading some targets
asset_df = web.DataReader('12_Industry_Portfolios', 'famafrench', start=start)[1]
asset_df.index = asset_df.index.to_timestamp()
assets = asset_df.columns.tolist()

# Loading some controls
control_df = pdr.get_data_famafrench("F-F_Research_Data_Factors", start=start)[0].div(100.0).drop(columns=["RF"])
control_df.index = control_df.index.to_timestamp()
controls = control_df.columns.tolist()

# Merge data into one dataframe aligned by date
df = pd.merge(factor_df, asset_df, left_index=True, right_index=True, how='left').ffill()
df = pd.merge(df, control_df, left_index=True, right_index=True, how='left').ffill()

# Running the rolling regression
ols = RollingOLS(window=12, expanding=False, lambda_=0.0, mode="joint")
ols.fit(factors=df[factors], controls=df[controls])
result = ols.transform(assets=df[assets])

# Plot some results
for f in factors:
    result.get_beta(f).plot(title=f)
```

See [`examples/fama_french_factors.ipynb`](https://github.com/GabinTB/rOLS/blob/main/examples/fama_french_factors.ipynb)
for a runnable, end-to-end version covering `mode`, full vs partial R², and
`estimate_every`.

---

## Index contract

Factors, controls, and targets must have indexes that are:

- **Identical** in length, labels, order, and type
- **Unique** (no duplicate timestamps)
- **Monotonically increasing**

Violations raise `ValueError` before any array conversion or estimation — rOLS
does not sort, deduplicate, reindex, join, or drop labels implicitly. Align
your inputs first:

```python
df = pd.concat([factors, controls, targets], axis=1).dropna()
```

---

## Result accessors

All results are indexed by time (rows) and target (columns):

```python
result.get_beta("f1")          # DataFrame (T × N_targets)
result.get_signal("f1")        # beta_t * factor_t (or lagged)
result.get_r2("f1")            # full-model R²
result.get_partial_r2("f1")    # f1's incremental R²
result.get_residuals("f1")     # endpoint regression residuals
result.get_fitted_values("f1") # full contemporaneous fitted value

result.get_se("f1")            # Newey-West SE — requires hac_lags
result.get_tstat("f1")         # beta / SE

result.get_control_beta("f1", "ctrl1")  # requires return_control_betas=True
result.get_dof("f1")           # residual degrees of freedom
result.get_n_used("f1")        # complete-case row count per endpoint
result.mode                    # "batched" or "joint"
```

For whole-panel work, iterate one factor at a time so inference frames can be
released as the loop advances:

```python
for factor, beta in result.iter_beta():
    process(factor, beta)

for factor, se in result.iter_se():
    process(factor, se)
```

---

## Long format output

```python
result.to_long("f1")                    # date, target, beta, signal, r2
result.to_long("f1", include_se=True)   # + se, t_stat
result.to_long_all()                    # all factors stacked
```
