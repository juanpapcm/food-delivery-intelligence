"""EDA for the Food Delivery Times dataset.

Run from the repo root:
    python model_pipeline/eda.py

Saves figures to outputs/figures/ and prints a summary of key findings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Food_Delivery_Times.csv"
FIG_DIR = REPO_ROOT / "outputs" / "figures"

TARGET = "Delivery_Time_min"
CATEGORICAL = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]


def save(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")


def iqr_outliers(s: pd.Series) -> tuple[float, float, pd.Series]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < low) | (s > high)
    return low, high, s[mask]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.relative_to(REPO_ROOT)}: {df.shape[0]} rows, {df.shape[1]} cols\n")

    # ---- 6. dtypes + missing values (up front, it informs the rest) --------
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    print("Dtypes & missing values:")
    print(summary.to_string())
    print()

    # ---- 1. Delivery_Time_min distribution ---------------------------------
    y = df[TARGET].dropna()
    desc = y.describe()
    print(f"{TARGET} stats:")
    print(desc.round(2).to_string())
    print(f"  skewness: {y.skew():.3f}   kurtosis: {y.kurtosis():.3f}\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(y, bins=30, kde=True, ax=ax, color="steelblue")
    ax.axvline(y.mean(), color="crimson", linestyle="--", label=f"mean={y.mean():.1f}")
    ax.axvline(y.median(), color="darkgreen", linestyle="--", label=f"median={y.median():.1f}")
    ax.set_title(f"Distribution of {TARGET}")
    ax.set_xlabel("minutes")
    ax.legend()
    save(fig, "01_target_distribution.png")

    # ---- 2. Distance vs delivery time + Pearson ----------------------------
    dd = df[["Distance_km", TARGET]].dropna()
    r, p = stats.pearsonr(dd["Distance_km"], dd[TARGET])
    print(f"Pearson(Distance_km, {TARGET}) = {r:.3f}  (p={p:.2e})\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(data=dd, x="Distance_km", y=TARGET, scatter_kws={"alpha": 0.4, "s": 25},
                line_kws={"color": "crimson"}, ax=ax)
    ax.set_title(f"Distance vs Delivery Time  (r = {r:.2f})")
    save(fig, "02_distance_vs_time.png")

    # ---- 3. Boxplots by categorical feature --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), CATEGORICAL):
        order = df.groupby(col)[TARGET].median().sort_values().index
        sns.boxplot(data=df, x=col, y=TARGET, order=order, ax=ax)
        ax.set_title(f"{TARGET} by {col}")
        ax.tick_params(axis="x", rotation=20)
    save(fig, "03_boxplots_categorical.png")

    # group means per category (printed for the console summary)
    print("Median delivery time by category:")
    for col in CATEGORICAL:
        med = df.groupby(col)[TARGET].median().sort_values()
        spread = med.max() - med.min()
        print(f"  {col:14s} spread={spread:5.1f} min   "
              f"slowest={med.idxmax()} ({med.max():.1f})   "
              f"fastest={med.idxmin()} ({med.min():.1f})")
    print()

    # ---- 4. Outlier detection with IQR -------------------------------------
    print("IQR outliers:")
    outlier_counts = {}
    for col in [TARGET, "Distance_km"]:
        low, high, out = iqr_outliers(df[col].dropna())
        outlier_counts[col] = len(out)
        print(f"  {col:18s} bounds=[{low:.2f}, {high:.2f}]   "
              f"{len(out)} outliers ({len(out)/len(df)*100:.1f}%)")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.boxplot(x=df[TARGET].dropna(), ax=axes[0], color="steelblue")
    axes[0].set_title(f"{TARGET} — IQR outliers")
    sns.boxplot(x=df["Distance_km"].dropna(), ax=axes[1], color="sandybrown")
    axes[1].set_title("Distance_km — IQR outliers")
    save(fig, "04_iqr_outliers.png")

    # ---- 5. Correlation heatmap (numeric) ----------------------------------
    num = df.select_dtypes(include=np.number).drop(columns=["Order_ID"], errors="ignore")
    corr = num.corr()
    print("Numeric correlations with target:")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())
    print()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Numeric correlation matrix")
    save(fig, "05_correlation_heatmap.png")

    # ---- Final summary ------------------------------------------------------
    top_cat = max(CATEGORICAL,
                  key=lambda c: df.groupby(c)[TARGET].median().max()
                              - df.groupby(c)[TARGET].median().min())
    top_cat_spread = (df.groupby(top_cat)[TARGET].median().max()
                      - df.groupby(top_cat)[TARGET].median().min())
    missing_cols = summary[summary["missing"] > 0].index.tolist()

    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"- Target mean={y.mean():.1f} min, median={y.median():.1f}, "
          f"skew={y.skew():.2f}  → {'right-skewed' if y.skew() > 0.3 else 'roughly symmetric'}.")
    print(f"- Distance is the strongest numeric driver (Pearson r = {r:.2f}).")
    print(f"- Biggest categorical effect: {top_cat} "
          f"(median spread ≈ {top_cat_spread:.1f} min across groups).")
    print(f"- IQR outliers: {outlier_counts[TARGET]} in {TARGET}, "
          f"{outlier_counts['Distance_km']} in Distance_km — keep an eye on them in modeling.")
    if missing_cols:
        print(f"- Columns with missing values: {', '.join(missing_cols)} — imputation needed.")
    else:
        print("- No missing values.")
    print(f"- Figures written to {FIG_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
