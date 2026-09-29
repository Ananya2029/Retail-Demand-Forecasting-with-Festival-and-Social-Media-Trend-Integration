"""
Leak-free retail demand forecasting pipeline (v2).

Predicts next week's sales for each of the 45 Walmart stores using only information
available before that week, and measures what festival and (simulated) social-media
features add on top of a strong baseline.

Run from the repository root:
    python pipeline/forecast_pipeline.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
import joblib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Walmart.csv"
PROCESSED = ROOT / "data" / "processed" / "final_featured_sales.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
TEST_WEEKS = 26
SEED = 42

# ---------------------------------------------------------------- festivals
# Walmart's data comes from US stores, so the festival calendar uses US retail events.
# Weights are domain priors for how strongly each event moves retail demand.
EVENTS = {
    "Super Bowl": (["2010-02-07", "2011-02-06", "2012-02-05", "2013-02-03"], 2),
    "Valentine's Day": (["2010-02-14", "2011-02-14", "2012-02-14", "2013-02-14"], 2),
    "Easter": (["2010-04-04", "2011-04-24", "2012-04-08", "2013-03-31"], 2),
    "Memorial Day": (["2010-05-31", "2011-05-30", "2012-05-28"], 2),
    "Independence Day": (["2010-07-04", "2011-07-04", "2012-07-04"], 2),
    "Back to School": (["2010-08-15", "2011-08-15", "2012-08-15"], 3),
    "Labor Day": (["2010-09-06", "2011-09-05", "2012-09-03"], 2),
    "Halloween": (["2010-10-31", "2011-10-31", "2012-10-31"], 2),
    "Thanksgiving": (["2010-11-25", "2011-11-24", "2012-11-22"], 4),
    "Black Friday": (["2010-11-26", "2011-11-25", "2012-11-23"], 5),
    "Christmas": (["2010-12-25", "2011-12-25", "2012-12-25"], 5),
    "New Year": (["2011-01-01", "2012-01-01", "2013-01-01"], 2),
}
CALENDAR = sorted(
    (pd.Timestamp(d), name, w) for name, (dates, w) in EVENTS.items() for d in dates
)


def festival_features(dates: pd.Series) -> pd.DataFrame:
    """Calendar-only features: known in advance, so they cannot leak the target."""
    rows = []
    for d in dates:
        week_end = d + pd.Timedelta(days=6)
        in_week = [(n, w) for e, n, w in CALENDAR if d <= e <= week_end]
        upcoming = [(e, n, w) for e, n, w in CALENDAR if e >= d]
        nxt = upcoming[0] if upcoming else (d + pd.Timedelta(days=365), "None", 0)
        days_to = (nxt[0] - d).days
        rows.append({
            "Festival_Flag": int(bool(in_week)),
            "Festival_Name": in_week[0][0] if in_week else "None",
            "Festival_Score": max((w for _, w in in_week), default=0),
            "Days_To_Festival": days_to,
            "Next_Festival_Score": nxt[2],
            "Pre_Festival_Flag": int(0 < days_to <= 14),
            # Demand build-up before big events: stronger and closer = higher
            "Festival_Proximity": nxt[2] * np.exp(-days_to / 14),
        })
    return pd.DataFrame(rows, index=dates.index)


# ---------------------------------------------------------------- social media
def simulated_social_features(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    SIMULATED social-media signals. No real social data is available for 2010-2012
    Walmart stores, so buzz is generated from the festival calendar plus store-level
    and week-to-week noise. It never uses sales. Values are published with a one-week
    lag (the model sees last week's buzz), as a real feed would be.
    """
    out = []
    for store, g in df.groupby("Store"):
        n = len(g)
        store_base = rng.normal(50, 8)
        ar = np.zeros(n)
        for i in range(1, n):
            ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 3)
        trend = store_base + 8 * g["Festival_Proximity"].to_numpy() + ar
        sentiment = np.clip(0.2 + 0.05 * g["Festival_Proximity"].to_numpy() + rng.normal(0, 0.15, n), -1, 1)
        mentions = np.maximum(0, trend * 60 + rng.normal(0, 300, n))
        s = pd.DataFrame({"Trend_Score": trend, "Sentiment_Score": sentiment,
                          "Mention_Count": mentions}, index=g.index)
        s[["Trend_Score_Lag1", "Sentiment_Score_Lag1", "Mention_Count_Lag1"]] = s.shift(1).to_numpy()
        out.append(s)
    return pd.concat(out).loc[df.index]


# ---------------------------------------------------------------- features
def build_dataset():
    df = pd.read_csv(RAW)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Sales history: every feature uses weeks strictly before the target week
    sales = df.groupby("Store")["Weekly_Sales"]
    df["Lag_1"] = sales.shift(1)
    df["Lag_2"] = sales.shift(2)
    df["Lag_52"] = sales.shift(52)                       # same week last year
    df["Rolling_Mean_4"] = sales.transform(lambda s: s.shift(1).rolling(4).mean())
    df["Rolling_Std_4"] = sales.transform(lambda s: s.shift(1).rolling(4).std())
    df["Rolling_Mean_12"] = sales.transform(lambda s: s.shift(1).rolling(12).mean())
    df["YoY_Ratio_Lag1"] = df["Lag_1"] / sales.shift(53)  # last week vs a year before it

    df = df.join(festival_features(df["Date"]))
    df = df.join(simulated_social_features(df, np.random.default_rng(SEED)))
    return df


BASE = ["Store", "WeekOfYear", "Month", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI",
        "Unemployment", "Lag_1", "Lag_2", "Lag_52", "Rolling_Mean_4", "Rolling_Std_4",
        "Rolling_Mean_12", "YoY_Ratio_Lag1"]
FESTIVAL = ["Festival_Flag", "Festival_Score", "Days_To_Festival", "Next_Festival_Score",
            "Pre_Festival_Flag", "Festival_Proximity"]
SOCIAL = ["Trend_Score_Lag1", "Sentiment_Score_Lag1", "Mention_Count_Lag1"]


def metrics(y, p):
    return {
        "MAE": round(mean_absolute_error(y, p), 0),
        "RMSE": round(root_mean_squared_error(y, p), 0),
        "WMAPE_%": round(100 * np.abs(y - p).sum() / np.abs(y).sum(), 2),
        "R2": round(r2_score(y, p), 4),
    }


def xgb():
    return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, random_state=SEED)


def main():
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    df = build_dataset()
    model_df = df.dropna(subset=BASE + SOCIAL).copy()

    cutoff = model_df["Date"].sort_values().unique()[-TEST_WEEKS]
    train = model_df[model_df["Date"] < cutoff]
    test = model_df[model_df["Date"] >= cutoff]
    y_tr, y_te = train["Weekly_Sales"], test["Weekly_Sales"]
    print(f"Train: {train.Date.min():%Y-%m-%d} – {train.Date.max():%Y-%m-%d} ({len(train)} rows) | "
          f"Test: {test.Date.min():%Y-%m-%d} – {test.Date.max():%Y-%m-%d} ({len(test)} rows, all 45 stores)")

    results = {
        "Naive (last week)": metrics(y_te, test["Lag_1"]),
        "Seasonal naive (same week last year)": metrics(y_te, test["Lag_52"]),
    }

    full = BASE + FESTIVAL + SOCIAL
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "XGBoost": xgb(),
    }
    preds = {}
    for name, m in models.items():
        m.fit(train[full], y_tr)
        preds[name] = m.predict(test[full])
        results[name] = metrics(y_te, preds[name])

    # Ablation: what does each feature group add? (XGBoost, same settings)
    ablation = {}
    for label, cols in [("Base (sales history + calendar + economy)", BASE),
                        ("+ Festival features", BASE + FESTIVAL),
                        ("+ Festival + simulated social", full)]:
        m = xgb().fit(train[cols], y_tr)
        ablation[label] = metrics(y_te, m.predict(test[cols]))

    res_df = pd.DataFrame(results).T
    abl_df = pd.DataFrame(ablation).T
    print("\nModel comparison (test = last 26 weeks):\n", res_df.to_string())
    print("\nFeature-group ablation (XGBoost):\n", abl_df.to_string())

    # Festival weeks specifically — where festival features should matter most
    fest_mask = test["Festival_Flag"].eq(1) | test["Pre_Festival_Flag"].eq(1)
    base_m = xgb().fit(train[BASE], y_tr)
    fest_m = xgb().fit(train[BASE + FESTIVAL], y_tr)
    fest_weeks = {
        "Base": metrics(y_te[fest_mask], base_m.predict(test.loc[fest_mask, BASE])),
        "+ Festival features": metrics(y_te[fest_mask], fest_m.predict(test.loc[fest_mask, BASE + FESTIVAL])),
    }
    print(f"\nFestival / pre-festival test weeks only ({fest_mask.sum()} rows):\n", pd.DataFrame(fest_weeks).T.to_string())

    best = min(models, key=lambda n: results[n]["MAE"])
    best_model = models[best]
    joblib.dump({"model": best_model, "features": full}, MODELS / "best_model.pkl")

    # SHAP on the test period
    sample = test[full].sample(min(1000, len(test)), random_state=SEED)
    sv = shap.TreeExplainer(best_model).shap_values(sample)
    shap_imp = pd.Series(np.abs(sv).mean(0), index=full).sort_values(ascending=False)
    shap_imp.rename("Importance").rename_axis("Feature").reset_index().to_csv(OUT / "shap_feature_importance.csv", index=False)

    # Outputs used by the dashboard and README
    res_df.to_csv(OUT / "model_comparison.csv")
    abl_df.to_csv(OUT / "ablation.csv")
    pred_df = test[["Date", "Store", "Weekly_Sales"]].assign(Predicted_Sales=preds[best].round(2))
    pred_df.to_csv(OUT / "predictions.csv", index=False)
    total = pred_df.groupby("Date")[["Weekly_Sales", "Predicted_Sales"]].sum().reset_index()
    total.to_csv(OUT / "forecast_results.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps({
        "test_period": [str(test.Date.min().date()), str(test.Date.max().date())],
        "best_model": best, "models": results, "ablation": ablation, "festival_weeks": fest_weeks,
    }, indent=2))
    df.drop(columns=["Trend_Score_Lag1", "Sentiment_Score_Lag1", "Mention_Count_Lag1"]).to_csv(PROCESSED, index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(total["Date"], total["Weekly_Sales"] / 1e6, label="Actual", color="black")
    ax.plot(total["Date"], total["Predicted_Sales"] / 1e6, label=f"{best} forecast", color="tab:red")
    ax.set_ylabel("Total weekly sales, 45 stores ($M)")
    ax.set_title("Test period: actual vs one-week-ahead forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "forecast_vs_actual.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap_imp.head(12).sort_values().plot.barh(ax=ax, color="tab:blue")
    ax.set_title(f"Mean |SHAP| — {best}")
    fig.tight_layout()
    fig.savefig(OUT / "shap_importance.png", dpi=120)
    print(f"\nBest model: {best}. Outputs written to {OUT}")


if __name__ == "__main__":
    main()
