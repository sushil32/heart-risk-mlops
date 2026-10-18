"""Exploratory data analysis: writes figures to reports/figures and a summary markdown file.

Usage:  python scripts/eda.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heartrisk.data import (  # noqa: E402
    BINARY,
    CATEGORICAL,
    CLEAN_PATH,
    NUMERIC,
    RAW_PATH,
    ROOT,
    TARGET,
    load_clean,
    load_raw,
)

OUT = ROOT / "reports" / "figures"
PALETTE = {0: "#00b894", 1: "#d63031"}
LABELS = {0: "No disease", 1: "Disease"}

CODE_NAMES = {
    "cp": {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abn.", 2: "LV hypertrophy"},
    "slope": {1: "up", 2: "flat", 3: "down"},
    "thal": {3: "normal", 6: "fixed", 7: "reversible"},
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "<=120", 1: ">120"},
    "exang": {0: "no", 1: "yes"},
}


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    raw = load_raw(RAW_PATH)
    df = load_clean(CLEAN_PATH)
    df["diagnosis"] = df[TARGET].map(LABELS)

    # 1. Missing values in the raw file
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3))
    miss.plot.bar(ax=ax, color="#fdcb6e")
    ax.set_title("Missing values per column (raw file)")
    ax.set_ylabel("count")
    save(fig, "eda_missing_values.png")

    # 2. Class balance
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = df["diagnosis"].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]][: len(counts)])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    save(fig, "eda_class_balance.png")

    # 3. Numeric distributions split by class
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUMERIC, strict=True):
        sns.histplot(data=df, x=col, hue=TARGET, palette=PALETTE, kde=True, ax=ax,
                     element="step", stat="density", common_norm=False)  # fmt: skip
        ax.legend(labels=[LABELS[1], LABELS[0]], fontsize=8)
    fig.suptitle("Numeric feature distributions by diagnosis", y=1.01)
    fig.tight_layout()
    save(fig, "eda_numeric_histograms.png")

    # 4. Boxplots (outliers)
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 3.8))
    for ax, col in zip(axes, NUMERIC, strict=True):
        sns.boxplot(data=df, x="diagnosis", y=col, hue="diagnosis", ax=ax,
                    palette={LABELS[k]: v for k, v in PALETTE.items()}, legend=False)  # fmt: skip
        ax.set_xlabel("")
    fig.suptitle("Outlier check by diagnosis", y=1.03)
    fig.tight_layout()
    save(fig, "eda_boxplots.png")

    # 5. Categorical / binary features: disease rate per category
    cats = CATEGORICAL + BINARY
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, cats, strict=False):
        rate = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        rate[col] = rate[col].map(lambda v, c=col: CODE_NAMES[c].get(int(v), v))
        ax.bar(rate[col].astype(str), rate["mean"], color="#e17055")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"], strict=True)):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Disease rate by {col}")
        ax.tick_params(axis="x", rotation=20)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save(fig, "eda_categorical_rates.png")

    # 6. Correlation heatmap
    corr = df[NUMERIC + CATEGORICAL + BINARY + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                cbar_kws={"shrink": 0.7}, ax=ax, annot_kws={"size": 8})  # fmt: skip
    ax.set_title("Spearman correlation matrix")
    save(fig, "eda_correlation_heatmap.png")

    # 7. Engineered feature preview: heart-rate reserve ratio
    hr = df.assign(hr_reserve_ratio=df["thalach"] / (220 - df["age"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=hr, x="hr_reserve_ratio", hue="diagnosis", fill=True, common_norm=False,
                palette={LABELS[k]: v for k, v in PALETTE.items()}, ax=ax)  # fmt: skip
    ax.set_title("Engineered: thalach / (220 - age)")
    save(fig, "eda_hr_reserve_ratio.png")

    # Text summary
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    summary = [
        "# EDA summary",
        "",
        f"- Raw rows: {len(raw)}, cleaned rows: {len(df)}",
        f"- Missing cells in raw data: {int(miss.sum())} "
        f"({', '.join(f'{k}={v}' for k, v in miss[miss > 0].items())})",
        f"- Class balance: {df[TARGET].value_counts().to_dict()} "
        f"(positive rate {df[TARGET].mean():.1%})",
        "",
        "## Strongest Spearman correlations with target",
        "",
        target_corr.round(3).to_frame("rho").to_markdown(),
        "",
        "## Numeric summary",
        "",
        df[NUMERIC].describe().T.round(2).to_markdown(),
    ]
    (ROOT / "reports" / "eda_summary.md").write_text("\n".join(summary) + "\n")
    print("  wrote reports/eda_summary.md")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
