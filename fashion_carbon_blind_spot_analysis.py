# ============================================================
# CASE #1 — FASHION'S CARBON BLIND SPOT
# Synthetic Fashion Supply Chain Emissions Analysis
# ============================================================
#
# Dataset description:
# This dataset is a modeled approximation of fashion supply chain
# emissions, based on industry benchmarks and publicly available
# LCA-style studies.
#
# Analytical goal:
# Transform a sustainability problem into a measurable operating
# system by connecting emissions, materials, logistics, sourcing,
# industry benchmarks, and KPI logic.
#
# Main thesis:
# Fashion brands often over-index on visible sustainability narratives
# such as materials, while the highest-leverage decarbonization levers
# sit inside operational decisions: manufacturing energy, planning,
# sourcing, and logistics.
#
# ============================================================


# ============================================================
## 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
## 2. DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

for folder in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CHART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
## 3. LOAD DATASET
# ============================================================

# Replace this path if your CSV has a different name.
# Recommended file: corrected_fashion_decarbonization_dataset_latest.csv

csv_path = RAW_DIR / "corrected_fashion_decarbonization_dataset_latest.csv"

df = pd.read_csv(csv_path)

print("Dataset loaded successfully.")
print(df.head())
print(df.info())


# ============================================================
## 4. STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("Standardized columns:")
print(df.columns.tolist())


# ============================================================
## 5. BASIC DATA CLEANING
# ============================================================

# Remove duplicate rows
df = df.drop_duplicates()

# Strip whitespace from text columns
text_cols = df.select_dtypes(include="object").columns

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

# Convert value column to numeric
if "value" in df.columns:
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Convert year column to numeric if available
if "year" in df.columns:
    df["year"] = (
        df["year"]
        .astype(str)
        .str.replace("2023-2025", "2025", regex=False)
        .str.replace("2023–2025", "2025", regex=False)
        .str.replace("2025E", "2025", regex=False)
        .str.replace("2026E", "2026", regex=False)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

print("Cleaning completed.")


# ============================================================
## 6. STANDARDIZE UNITS
# ============================================================

unit_map = {
    "%": "percent",
    "percentage": "percent",
    "pct": "percent",
    "kg co2e": "kgCO2e",
    "kg co₂e": "kgCO2e",
    "kgco2e": "kgCO2e",
    "tco2e": "tCO2e",
    "tonnes co2e": "tCO2e",
    "tons co2e": "tCO2e",
    "$": "USD",
    "usd": "USD"
}

if "unit" in df.columns:
    df["unit"] = df["unit"].str.lower().map(
        lambda x: unit_map.get(x, x) if pd.notna(x) else x
    )

print("Units standardized.")


# ============================================================
## 7. CORRECT KNOWN SEMANTIC ISSUES
# ============================================================

# Some variables can be renamed to better reflect their analytical meaning.
variable_map = {
    "recycled_material_share": "sustainable_material_share",
    "efficiency_priority": "cost_efficiency_priority_share",
    "executives_trade_disruption": "executives_disrupted_trade_flows_risk_share",
}

if "variable" in df.columns:
    df["variable"] = df["variable"].replace(variable_map)

# Correct benchmark interpretation:
# 40% = executives citing disrupted trade flows / deglobalisation as a top-three growth risk.
# 76% = broader benchmark about trade disruptions and rising duties shaping the industry.
if {"variable", "value"}.issubset(df.columns):
    mask = df["variable"].eq("executives_disrupted_trade_flows_risk_share")
    df.loc[mask, "value"] = 40.0

    if "notes" in df.columns:
        df.loc[mask, "notes"] = (
            "Corrected: 40% = executives citing disrupted trade flows/deglobalisation "
            "as a top-three growth risk. 76% refers more broadly to trade disruptions "
            "and rising duties shaping the industry."
        )

print("Known semantic corrections applied.")


# ============================================================
## 8. ADD ANALYTICAL CLASSIFICATIONS
# ============================================================

def classify_data_type(row):
    variable = str(row.get("variable", "")).lower()

    if any(term in variable for term in ["share", "percent", "priority", "risk"]):
        return "benchmark_share"

    if any(term in variable for term in ["emissions", "co2", "carbon"]):
        return "emissions_estimate"

    if any(term in variable for term in ["material", "sustainable", "recycled"]):
        return "material_metric"

    return "context_metric"


if "data_type" not in df.columns:
    df["data_type"] = df.apply(classify_data_type, axis=1)

if "modeling_confidence" not in df.columns:
    df["modeling_confidence"] = "medium"

print("Analytical classifications added.")


# ============================================================
## 9. DATA QUALITY CHECKS
# ============================================================

quality_checks = {}

quality_checks["rows"] = len(df)
quality_checks["columns"] = len(df.columns)
quality_checks["duplicates"] = df.duplicated().sum()
quality_checks["missing_values_total"] = df.isna().sum().sum()

if "value" in df.columns:
    quality_checks["negative_values"] = (df["value"] < 0).sum()
else:
    quality_checks["negative_values"] = None

quality_summary = pd.DataFrame(
    list(quality_checks.items()),
    columns=["check", "result"]
)

quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

print("Data quality summary:")
print(quality_summary)


# ============================================================
## 10. SAVE CLEANED DATASET
# ============================================================

cleaned_path = PROCESSED_DIR / "fashion_decarbonization_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_path}")


# ============================================================
## 11. CREATE ANALYTICAL TABLES
# ============================================================

# Table 1: overview by impact area
if {"impact_area", "value"}.issubset(df.columns):
    impact_summary = (
        df.groupby("impact_area", dropna=False)["value"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    impact_summary.to_csv(OUTPUT_DIR / "impact_area_summary.csv", index=False)
    print("Impact area summary exported.")

# Table 2: overview by category
if {"category", "value"}.issubset(df.columns):
    category_summary = (
        df.groupby("category", dropna=False)["value"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    category_summary.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)
    print("Category summary exported.")


# ============================================================
## 12. DEFINE VISUAL STYLE
# ============================================================

# Palette based on the project visual system
COLORS = {
    "manufacturing": "#d0021b",
    "materials": "#5a0e4d",
    "logistics": "#f2b6aa",
    "text": "#370707",
    "accent": "#d94383",
    "background": "#ffffff",
    "grid": "#dddddd"
}

plt.rcParams.update({
    "figure.facecolor": COLORS["background"],
    "axes.facecolor": COLORS["background"],
    "savefig.facecolor": COLORS["background"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "axes.edgecolor": COLORS["text"],
    "font.size": 11
})


# ============================================================
## 13. VISUALIZATION 1 — EMISSIONS CONCENTRATION BY LEVER
# ============================================================

# This chart represents the core case thesis:
# Manufacturing carries the highest emissions concentration,
# while materials often receive more public/brand attention.

emissions_levers = pd.DataFrame({
    "lever": ["Manufacturing", "Materials", "Logistics"],
    "share": [64, 30, 6]
})

lever_colors = [
    COLORS["manufacturing"],
    COLORS["materials"],
    COLORS["logistics"]
]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    emissions_levers["lever"],
    emissions_levers["share"],
    color=lever_colors
)

for bar, value in zip(bars, emissions_levers["share"]):
    ax.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value}%",
        va="center",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"]
    )

ax.set_title(
    "Emissions Concentration by Lever",
    fontsize=18,
    fontweight="bold",
    color=COLORS["text"]
)

ax.set_xlabel("Relative emissions share (%)")
ax.set_xlim(0, 75)
ax.grid(axis="x", alpha=0.25, color=COLORS["grid"])

plt.tight_layout()
plt.savefig(CHART_DIR / "01_emissions_concentration_by_lever.png", dpi=300)
plt.close()

print("Chart generated: emissions concentration by lever.")


# ============================================================
## 14. VISUALIZATION 2 — ATTENTION VS IMPACT MATRIX
# ============================================================

# This matrix shows the blind spot:
# materials are often highly visible, while manufacturing has higher leverage.

matrix_data = pd.DataFrame({
    "lever": ["Manufacturing", "Materials", "Logistics"],
    "attention": [3, 8, 6],
    "impact": [8, 7, 3],
    "label": [
        "High impact\nLow attention",
        "High attention\nMedium impact",
        "Medium attention\nLow impact"
    ]
})

color_map = {
    "Manufacturing": COLORS["manufacturing"],
    "Materials": COLORS["materials"],
    "Logistics": COLORS["logistics"]
}

fig, ax = plt.subplots(figsize=(8, 7))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.axhline(5, color=COLORS["text"], linestyle="--", linewidth=1, alpha=0.55)
ax.axvline(5, color=COLORS["text"], linestyle="--", linewidth=1, alpha=0.55)

for _, row in matrix_data.iterrows():
    ax.scatter(
        row["attention"],
        row["impact"],
        s=900,
        color=color_map[row["lever"]],
        edgecolor="none"
    )

    ax.text(
        row["attention"] + 0.45,
        row["impact"],
        f"{row['lever']}\n{row['label']}",
        va="center",
        ha="left",
        fontsize=10,
        color="#000000"
    )

ax.set_title(
    "Attention vs Impact",
    fontsize=18,
    fontweight="bold",
    color=COLORS["text"]
)

ax.set_xlabel("Brand attention / visibility")
ax.set_ylabel("Impact / leverage")

ax.text(0, -0.7, "Low", fontsize=10, color=COLORS["text"])
ax.text(9.2, -0.7, "High", fontsize=10, color=COLORS["text"])
ax.text(-0.9, 0, "Low", fontsize=10, color=COLORS["text"], rotation=90)
ax.text(-0.9, 9.2, "High", fontsize=10, color=COLORS["text"], rotation=90)

plt.tight_layout()
plt.savefig(CHART_DIR / "02_attention_vs_impact_matrix.png", dpi=300)
plt.close()

print("Chart generated: attention vs impact matrix.")


# ============================================================
## 15. VISUALIZATION 3 — PRIORITIZATION TABLE
# ============================================================

# This table translates the analysis into a decision framework.
# It evaluates initiatives using Impact × Effort × Speed.

prioritization = pd.DataFrame({
    "initiative": [
        "Supplier renewable energy",
        "Reduce air freight",
        "Production forecasting",
        "Low-carbon sourcing"
    ],
    "impact": ["High", "Medium", "High", "Medium"],
    "effort": ["High", "Medium", "Medium", "High"],
    "speed": ["Medium", "Fast", "Medium", "Slow"]
})

prioritization.to_csv(OUTPUT_DIR / "prioritization_table.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")

table = ax.table(
    cellText=prioritization.values,
    colLabels=["Initiative", "Impact", "Effort", "Speed"],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor(COLORS["materials"])

    if row == 0:
        cell.set_facecolor(COLORS["manufacturing"])
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#ffffff")
        cell.set_text_props(color=COLORS["text"])

ax.set_title(
    "Prioritization Lens: Impact × Effort × Speed",
    fontsize=16,
    fontweight="bold",
    color=COLORS["text"],
    pad=20
)

plt.tight_layout()
plt.savefig(CHART_DIR / "03_prioritization_table.png", dpi=300)
plt.close()

print("Chart generated: prioritization table.")


# ============================================================
## 16. VISUALIZATION 4 — KPI SCORECARD
# ============================================================

# The KPI plan turns sustainability into a measurable operating system.
# Leading indicators track behavior change.
# Lagging indicators track business/climate outcomes.

leading_indicators = [
    "% production sourced from low-carbon suppliers",
    "% orders shipped by air",
    "Lead-time adherence",
    "Forecast accuracy",
    "Renewable energy penetration in Tier 1/2 suppliers"
]

lagging_indicators = [
    "CO₂e per unit",
    "CO₂e per revenue",
    "Total Scope 3 reduction by category",
    "Manufacturing emissions intensity"
]

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

ax.text(
    0.15, 0.9,
    "Leading indicators",
    fontsize=17,
    fontweight="bold",
    color=COLORS["manufacturing"]
)

ax.text(
    0.62, 0.9,
    "Lagging indicators",
    fontsize=17,
    fontweight="bold",
    color=COLORS["materials"]
)

for i, indicator in enumerate(leading_indicators):
    ax.text(
        0.1,
        0.75 - i * 0.12,
        f"• {indicator}",
        fontsize=12,
        color=COLORS["text"]
    )

for i, indicator in enumerate(lagging_indicators):
    ax.text(
        0.58,
        0.75 - i * 0.12,
        f"• {indicator}",
        fontsize=12,
        color=COLORS["text"]
    )

ax.axvline(0.5, ymin=0.15, ymax=0.85, color=COLORS["materials"], alpha=0.45)

ax.set_title(
    "KPI Plan: Measuring Sustainability Like Performance",
    fontsize=17,
    fontweight="bold",
    color=COLORS["text"],
    pad=20
)

plt.tight_layout()
plt.savefig(CHART_DIR / "04_kpi_scorecard.png", dpi=300)
plt.close()

print("Chart generated: KPI scorecard.")


# ============================================================
## 17. STRATEGIC INSIGHTS
# ============================================================

insights = [
    {
        "insight": "Manufacturing is the highest-leverage decarbonization lever.",
        "interpretation": (
            "Energy transition and process efficiency in supplier operations "
            "should be prioritized before purely communication-led sustainability actions."
        )
    },
    {
        "insight": "Materials matter, but they can receive disproportionate attention.",
        "interpretation": (
            "Material sourcing is important, but it is often more visible and easier "
            "to communicate than manufacturing transformation."
        )
    },
    {
        "insight": "Logistics is a smaller emissions share but a critical planning signal.",
        "interpretation": (
            "Air freight often acts as a patch for weak forecasting, compressed lead times, "
            "or poor SKU prioritization."
        )
    },
    {
        "insight": "Decarbonization requires an operating system.",
        "interpretation": (
            "Progress depends on KPIs, decision cadence, supplier scorecards, "
            "accountability, and operational discipline."
        )
    }
]

insights_df = pd.DataFrame(insights)
insights_df.to_csv(OUTPUT_DIR / "strategic_insights.csv", index=False)

print("Strategic insights exported.")


# ============================================================
## 18. ROADMAP RECOMMENDATIONS
# ============================================================

roadmap = pd.DataFrame({
    "pillar": [
        "Manufacturing first",
        "Demand planning to reduce air freight",
        "Carbon-informed sourcing",
        "Carbon performance operating system"
    ],
    "action": [
        "Supplier energy transition + manufacturing efficiency",
        "Improve forecasting, lead-time buffers, and SKU prioritization rules",
        "Embed carbon into vendor scorecards and buying decisions",
        "Treat sustainability like performance, with KPIs, cadence, and accountability"
    ],
    "expected_result": [
        "Lower production-related emissions",
        "Reduced emergency air freight and logistics emissions",
        "Better sourcing decisions across cost, speed, and carbon",
        "Continuous monitoring of operational decarbonization"
    ]
})

roadmap.to_csv(OUTPUT_DIR / "recommendations_roadmap.csv", index=False)

print("Roadmap recommendations exported.")


# ============================================================
## 19. GENERATE TEXT SUMMARY
# ============================================================

summary_text = """
CASE #1 — FASHION'S CARBON BLIND SPOT

This analysis reframes fashion sustainability as an operational performance challenge.
The key finding is that the greatest emissions leverage sits in manufacturing, especially
supplier energy and process efficiency, while much of the public sustainability narrative
focuses on materials.

The analysis maps the fashion value chain into three controllable levers:
manufacturing, materials, and logistics. It then compares attention versus impact to
surface the central blind spot: the areas that receive the most visibility are not always
the areas that create the most decarbonization leverage.

The recommended roadmap prioritizes:
1. Supplier renewable energy and manufacturing efficiency.
2. Demand planning to reduce air freight dependency.
3. Carbon-informed sourcing and vendor scorecards.
4. A KPI-driven measurement system.

Conclusion:
The competitive advantage is not looking sustainable. It is building a lower-carbon
operating model.
"""

summary_path = OUTPUT_DIR / "case_summary.txt"
summary_path.write_text(summary_text, encoding="utf-8")

print("Text summary exported.")


# ============================================================
## 20. FINAL OUTPUT CHECK
# ============================================================

print("\nAnalysis complete.")
print("\nGenerated files:")
print(f"- Cleaned dataset: {cleaned_path}")
print(f"- Data quality summary: {OUTPUT_DIR / 'data_quality_summary.csv'}")
print(f"- Impact area summary: {OUTPUT_DIR / 'impact_area_summary.csv'}")
print(f"- Category summary: {OUTPUT_DIR / 'category_summary.csv'}")
print(f"- Prioritization table: {OUTPUT_DIR / 'prioritization_table.csv'}")
print(f"- Strategic insights: {OUTPUT_DIR / 'strategic_insights.csv'}")
print(f"- Roadmap: {OUTPUT_DIR / 'recommendations_roadmap.csv'}")
print(f"- Summary text: {summary_path}")
print(f"- Charts folder: {CHART_DIR}")


# ============================================================
# END OF SCRIPT
# ============================================================
