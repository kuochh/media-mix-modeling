# %% [markdown]
# # Phase 1: EDA and Setup
# Load raw Conjura data, aggregate to df_weekly, profile brands, select modeling candidate.

# %% Step 1: Load and inspect raw data
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots

df = pd.read_csv("data/raw/conjura_mmm_data.csv")
df.columns = df.columns.str.lower()
df["date_day"] = pd.to_datetime(df["date_day"])
mask = df["territory_name"] == "US"
df = df[mask]
print(f"Shape: {df.shape}")

# %% Step 2: Profile brands at a glance
cols_spend = [c for c in df.columns if c.endswith("_spend")]

df_grain = (
    df.groupby("organisation_id")
    .agg(
        vertical=("organisation_vertical", "first"),
        date_min=("date_day", "min"),
        date_max=("date_day", "max"),
        n_days=("date_day", "nunique"),
        # all purchases
        mean_all_purchases=("all_purchases", "mean"),
        std_all_purchases=("all_purchases", "std"),
        mean_all_purchases_units=("all_purchases_units", "mean"),
        std_all_purchases_units=("all_purchases_units", "std"),
        # first purchases
        mean_first_purchases=("first_purchases", "mean"),
        std_first_purchases=("first_purchases", "std"),
        mean_first_purchases_units=("first_purchases_units", "mean"),
        std_first_purchases_units=("first_purchases_units", "std"),
    )
)

# Compute spend and channel counts from the grouped data
for org_id in df_grain.index:
    org_data = df.loc[df["organisation_id"] == org_id, cols_spend]
    df_grain.loc[org_id, "total_spend"] = org_data.sum().sum()
    df_grain.loc[org_id, "n_active_channels"] = (org_data.sum() > 0).sum()

df_grain = df_grain.sort_values("total_spend", ascending=False)

print(f"US brands: {len(df_grain)}")
df_grain.head(100)


# %% [markdown]
# ## Why these two candidates?
#
# 10 US brands in the dataset. Filtered to higher-spending brands for modeling
# viability. Top 5 by total spend: #1 food & drink, #2-3 apparel, #4-5 business
# & industrial. Selected the two apparel brands (apparel_1, apparel_2) as
# candidates because they share a vertical (controlled comparison) and both have
# substantial multi-channel spend.

# %% Step 2b: Inspect candidate brands
candidates = {
    "apparel_1": "784d6aa3cda59f59f2400332b2420a49",
    "apparel_2": "4a762f02ca755b22d37393e8dbeab1a6",
}

for label, org_id in candidates.items():
    brand = df[df["organisation_id"] == org_id]
    dates = brand["date_day"].sort_values()

    # Date gaps
    gaps = dates.diff().dt.days
    max_gap = gaps.max()
    n_gaps = (gaps > 1).sum()

    # Per-channel spend
    channel_spend = brand[cols_spend].sum().sort_values(ascending=False)
    active = channel_spend[channel_spend > 0]

    print(f"\n--- {label} ({org_id[:8]}...) ---")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Days: {len(dates)}, Gaps >1 day: {n_gaps}, Max gap: {max_gap} days")
    print(f"Active channels ({len(active)}/9):")
    for ch, spend in active.items():
        pct_nonzero = (brand[ch] > 0).mean() * 100
        print(f"  {ch}: ${spend:,.0f} total, {pct_nonzero:.0f}% of days nonzero")


# %% [markdown]
# ## Brand selection
#
# **apparel_2** is the primary modeling brand. 9 active spend channels with 4
# carrying meaningful, consistent spend: Meta Facebook (100% nonzero), Google
# Paid Search (95%), Google PMax (72%), Meta Instagram (65%). This gives a real
# Google-vs-Meta allocation story ($907K Google vs. $1.67M Meta), which is
# critical for budget optimization to have something to say. 147 weeks
# (~2.8 years), zero date gaps, USD currency.
#
# **apparel_1** rejected because spend is too concentrated in Meta
# Facebook/Instagram ($5.8M of $6.3M total). Google channels have under 15%
# nonzero days. Not enough variation to identify channel-level effects across
# platforms.
#
# **Spend** (not impressions) used as the media input variable — the conventional
# choice in MMM practice (PyMC-Marketing, LightweightMMM, Robyn all default to
# spend). The tradeoff: spend conflates CPM variation with volume; impressions
# capture reach more directly. Documented but does not change the decision here.

# %% Step 3: Filter to selected brands and aggregate to df_weekly
selected_orgs = [
    "784d6aa3cda59f59f2400332b2420a49",  # apparel_1
    "4a762f02ca755b22d37393e8dbeab1a6",  # apparel_2
]
df = df[df["organisation_id"].isin(selected_orgs)]

df["iso_year"] = df["date_day"].dt.isocalendar().year.astype(int)
df["iso_week"] = df["date_day"].dt.isocalendar().week.astype(int)

# Metadata cols are constant within a timeseries — take first
cols_meta = [
    "organisation_id", "organisation_vertical", "organisation_subvertical",
    "organisation_marketing_sources", "organisation_primary_territory_name",
    "territory_name", "currency_code",
]

cols_clicks = [c for c in df.columns if c.endswith("_clicks")]
cols_impressions = [c for c in df.columns if c.endswith("_impressions")]
cols_outcome = [c for c in df.columns if c.startswith(("all_purchases", "first_purchases"))]
cols_numeric = cols_spend + cols_clicks + cols_impressions + cols_outcome
agg_dict = {c: "sum" for c in cols_numeric}
agg_dict["date_day"] = "count"
agg_dict.update({c: "first" for c in cols_meta})

df_weekly = (
    df.groupby(["mmm_timeseries_id", "iso_year", "iso_week"])
    .agg(agg_dict)
    .rename(columns={"date_day": "n_days"})
    .reset_index()
)

# Drop partial weeks
n_before = len(df_weekly)
df_weekly = df_weekly[df_weekly["n_days"] == 7]
print(f"Dropped {n_before - len(df_weekly)} partial weeks")
print(f"Daily shape:  {df.shape}")
print(f"Weekly shape: {df_weekly.shape}")
df_weekly.head(5)


# %% [markdown]
# ## Aggregation notes
#
# Dropped 2 partial weeks (first and last weeks with fewer than 7 days). Partial
# weeks would distort weekly totals, especially spend and purchase sums, making
# them incomparable to full weeks. Both brands aggregated to keep the pipeline
# general for potential cross-brand comparison in Act 3, even though apparel_2 is
# the primary modeling brand.

# %% Step 4: Plot outcome variables for apparel_2
# Reconstruct week-start date from ISO year/week
df_weekly["week_start"] = pd.to_datetime(
    df_weekly["iso_year"].astype(str)
    + "-W"
    + df_weekly["iso_week"].astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
)
df_weekly = df_weekly.sort_values(["organisation_id", "week_start"]).reset_index(
    drop=True
)

ORG_APPAREL_2 = "4a762f02ca755b22d37393e8dbeab1a6"
df_a2 = df_weekly[df_weekly["organisation_id"] == ORG_APPAREL_2]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
fig.add_trace(
    go.Scatter(x=df_a2["week_start"], y=df_a2["all_purchases"], name="All Purchases"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_a2["week_start"], y=df_a2["first_purchases"], name="First Purchases"
    ),
    row=2, col=1,
)
fig.update_layout(title="apparel_2: Weekly Purchases", height=500)
fig.show()


# %% Step 4b: Weekly spend by channel
cols_spend_a2 = [c for c in cols_spend if df_a2[c].sum() > 0]

fig = go.Figure()
for ch in cols_spend_a2:
    fig.add_trace(
        go.Scatter(x=df_a2["week_start"], y=df_a2[ch], name=ch.replace("_spend", ""))
    )
fig.update_layout(title="apparel_2: Weekly Spend by Channel", height=500)
fig.show()


# %% Step 4c: Zero-inflation table
rows = []
for ch in cols_spend_a2:
    total = df_a2[ch].sum()
    pct_zero = (df_a2[ch] == 0).mean() * 100
    nonzero = df_a2.loc[df_a2[ch] > 0, ch]
    rows.append({
        "channel": ch.replace("_spend", ""),
        "total_spend": total,
        "pct_zero_weeks": round(pct_zero, 1),
        "mean_nonzero": round(nonzero.mean(), 2) if len(nonzero) > 0 else 0,
        "median_nonzero": round(nonzero.median(), 2) if len(nonzero) > 0 else 0,
    })
df_zero = pd.DataFrame(rows).sort_values("total_spend", ascending=False)
df_zero # type: ignore


# %% [markdown]
# ## Channel assessment
#
# 4 channels have clearly consistent spend: Meta Facebook (100% nonzero weeks),
# Google Paid Search (95%), Google PMax (72%), Meta Instagram (65%). Google
# Shopping (41% nonzero, $70K total) is borderline -- sparse but not negligible,
# and borderline significant in OLS (p=0.098). Worth including in the Bayesian
# model and letting the posterior decide. The rest are effectively
# unidentifiable: TikTok (6% nonzero), Google Display (4%), Google Video (1%),
# meta_other negligible. All channels will be included in the model, but the
# sparse ones should get wide posteriors, which is the correct behavior.
#
# Notable patterns in the spend time series: a large spike in Meta Facebook
# spend at the beginning and end of the data. The late spike coincides with a
# spike in purchases. All purchases and first purchases track almost
# identically, suggesting limited repeat-purchase signal in this brand. The
# discount columns in the dataset might help separate acquisition from
# promotion-driven revenue.

# %% Step 5: Create control variables
# Time index per brand (data already sorted by org + week_start)
df_weekly["t"] = df_weekly.groupby("organisation_id").cumcount()

# Fourier features: 2 harmonic pairs at yearly frequency (52-week period)
for k in [1, 2]:
    df_weekly[f"sin_{k}"] = np.sin(2 * np.pi * k * df_weekly["t"] / 52)
    df_weekly[f"cos_{k}"] = np.cos(2 * np.pi * k * df_weekly["t"] / 52)

# Non-paid traffic aggregate
cols_nonpaid = [
    "direct_clicks", "branded_search_clicks", "organic_search_clicks",
    "email_clicks", "referral_clicks", "all_other_clicks",
]
df_weekly["nonpaid_clicks"] = df_weekly[cols_nonpaid].sum(axis=1)

print("Control columns added: t, sin_1, cos_1, sin_2, cos_2, nonpaid_clicks")
df_weekly[["week_start", "t", "sin_1", "cos_1", "sin_2", "cos_2", "nonpaid_clicks"]].head(10)


# %% [markdown]
# ## Control feature choices
#
# 2 Fourier harmonic pairs at a 52-week period. The first harmonic captures the
# main annual cycle; the second captures asymmetry (e.g., a sharper holiday peak
# vs. a gradual summer trough). A third harmonic would model ~17-week
# sub-cycles, which risks overfitting on only 147 weeks of data. Non-paid
# traffic clicks aggregated into a single feature rather than kept as 6 separate
# regressors. With 147 observations and 9 spend channels already in the model,
# splitting nonpaid into separate channels would eat degrees of freedom for
# variables that are not the focus of the analysis.

# %% Step 6: Naive OLS
df_ols = df_weekly[df_weekly["organisation_id"] == ORG_APPAREL_2].copy()

# Only include spend channels with nonzero total
cols_spend_active = [c for c in cols_spend if df_ols[c].sum() > 0]
print(f"Active spend channels: {cols_spend_active}")

y = df_ols["all_purchases"]

# Model A: spend channels only
X_a = sm.add_constant(df_ols[cols_spend_active])
model_a = sm.OLS(y, X_a).fit()
print(f"\n--- Model A: Spend only (R² = {model_a.rsquared:.3f}) ---")
print(model_a.summary().tables[1])

# Model B: spend + trend + seasonality
cols_controls = ["t", "sin_1", "cos_1", "sin_2", "cos_2"]
X_b = sm.add_constant(df_ols[cols_spend_active + cols_controls])
model_b = sm.OLS(y, X_b).fit()
print(f"\n--- Model B: Spend + trend/seasonality (R² = {model_b.rsquared:.3f}) ---")
print(model_b.summary().tables[1])

# Model C: + all non-paid traffic
X_c = sm.add_constant(df_ols[cols_spend_active + cols_controls + ["nonpaid_clicks"]])
model_c = sm.OLS(y, X_c).fit()
print(f"\n--- Model C: + all nonpaid_clicks (R² = {model_c.rsquared:.3f}) ---")
print(model_c.summary().tables[1])

# Model D: + exogenous-only non-paid traffic (exclude branded_search, direct)
cols_exog_nonpaid = [
    "organic_search_clicks", "email_clicks", "referral_clicks", "all_other_clicks",
]
df_ols["exog_nonpaid_clicks"] = df_ols[cols_exog_nonpaid].sum(axis=1)
X_d = sm.add_constant(df_ols[cols_spend_active + cols_controls + ["exog_nonpaid_clicks"]])
model_d = sm.OLS(y, X_d).fit()
print(f"\n--- Model D: + exog nonpaid only (R² = {model_d.rsquared:.3f}) ---")
print(model_d.summary().tables[1])


# %% Step 6b: OLS model comparison
df_compare = pd.DataFrame([
    {"model": "A", "regressors": "spend only", "R²": model_a.rsquared, "adj_R²": model_a.rsquared_adj},
    {"model": "B", "regressors": "+ trend + seasonality", "R²": model_b.rsquared, "adj_R²": model_b.rsquared_adj},
    {"model": "C", "regressors": "+ all nonpaid_clicks", "R²": model_c.rsquared, "adj_R²": model_c.rsquared_adj},
    {"model": "D", "regressors": "+ exog nonpaid only", "R²": model_d.rsquared, "adj_R²": model_d.rsquared_adj},
])
df_compare[["R²", "adj_R²"]] = df_compare[["R²", "adj_R²"]].round(3)
df_compare # type: ignore


# %% [markdown]
# ## OLS takeaways
#
# R² is high across all specs (0.914 to 0.939), but this should not be mistaken
# for a good model. OLS here is capturing correlations between spend and
# purchases with no adstock (carryover effects) and no saturation (diminishing
# returns). The coefficients are not causal estimates and should not be used for
# budget decisions. Several channels show nonsensical coefficients (meta_other
# at -26, negative TikTok) driven by collinearity and sparse data.
#
# What the OLS exercise does confirm: (1) spend and purchases move together,
# which is a necessary baseline for modeling; (2) seasonality is real but
# incremental (R² jumps only 0.018 from Model A to B, first Fourier harmonic is
# significant); (3) nonpaid traffic adds modest explanatory power (R² 0.932 to
# 0.939). This is a sanity check, not a model to make decisions from. The
# Bayesian model in Phase 2 adds adstock and saturation, which will change the
# coefficient interpretation entirely.

# %% Step 7: Save to data/processed/
os.makedirs("data/processed", exist_ok=True)

ORG_APPAREL_1 = "784d6aa3cda59f59f2400332b2420a49"

for label, org_id in [("apparel_1", ORG_APPAREL_1), ("apparel_2", ORG_APPAREL_2)]:
    df_brand = df_weekly[df_weekly["organisation_id"] == org_id]
    path = f"data/processed/{label}.csv"
    df_brand.to_csv(path, index=False)
    print(f"Saved {label}: {df_brand.shape} → {path}")