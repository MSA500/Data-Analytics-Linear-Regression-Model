import pandas as pd
import numpy as np
from fastapi import APIRouter
from scipy import stats

router = APIRouter()

# ─────────────────────────────────────────────
# Load + Clean + Encode (same as stats.py)
# ─────────────────────────────────────────────
df_raw = pd.read_csv("data/dataSet.csv")
df = df_raw.copy()
df.dropna(inplace=True)
df = df[df["CGPA"].between(0, 10)]
df.reset_index(drop=True, inplace=True)
df["Placement"]             = df["Placement"].map({"Yes": 1, "No": 0})
df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})
df["CGPA"]                  = ((df["CGPA"] / 10) * 4).round(2)

NUMERIC_COLS = [
    "IQ", "Prev_Sem_Result", "CGPA",
    "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed",
]


# ─────────────────────────────────────────────
# ENDPOINT 1 — /api/probability/normality
# Shapiro-Wilk normality test for each column
# + KDE curve data points for histogram overlay
# ─────────────────────────────────────────────
@router.get("/probability/normality")
def normality_tests():
    result = {}

    for col in NUMERIC_COLS:
        data   = df[col].dropna()
        sample = data.sample(min(5000, len(data)), random_state=42)

        # Shapiro-Wilk test (best for n < 5000)
        stat, p_value = stats.shapiro(sample)

        # Normal distribution fit
        mu, sigma = float(data.mean()), float(data.std())

        # KDE curve — 60 evenly spaced points across data range
        x_min = float(data.min())
        x_max = float(data.max())
        x_pts = np.linspace(x_min, x_max, 60)

        # Actual KDE (smoothed density from real data)
        kde      = stats.gaussian_kde(data)
        kde_vals = kde(x_pts)

        # Fitted normal PDF curve
        norm_vals = stats.norm.pdf(x_pts, mu, sigma)

        result[col] = {
            "shapiro_stat":   round(float(stat), 4),
            "p_value":        round(float(p_value), 4),
            # If p > 0.05 → data looks normal, else not normal
            "is_normal":      bool(p_value > 0.05),
            "interpretation": "Normally distributed (p > 0.05)" if p_value > 0.05
                              else "Not normally distributed (p ≤ 0.05)",
            "mean":           round(mu, 4),
            "std":            round(sigma, 4),
            "kde_curve": [
                {"x": round(float(x), 4), "kde": round(float(k), 6), "normal": round(float(n), 6)}
                for x, k, n in zip(x_pts, kde_vals, norm_vals)
            ],
        }

    return result


# ─────────────────────────────────────────────
# ENDPOINT 3 — /api/probability/calculate
# Probability calculator:
# P(variable > threshold) and P(variable < threshold)
# Query params: col, threshold
# ─────────────────────────────────────────────
@router.get("/probability/calculate")
def calculate_probability(col: str = "CGPA", threshold: float = 3.0):

    if col not in NUMERIC_COLS:
        return {"error": f"Column '{col}' not available. Choose from {NUMERIC_COLS}"}

    data = df[col].dropna()
    mu = float(data.mean())
    sigma = float(data.std())
    n = len(data)

    emp_above = round(float((data > threshold).mean()) * 100, 2)
    emp_below = round(float((data < threshold).mean()) * 100, 2)

    above_threshold = df[df[col] > threshold]

    if len(above_threshold) > 0:
        cond_placement = round(float(above_threshold["Placement"].mean()) * 100, 2)
    else:
        cond_placement = 0.0

    if col == "IQ":
        z_score = round((threshold - mu) / sigma, 4) if sigma != 0 else 0.0
        theo_below = round(float(stats.norm.cdf(threshold, mu, sigma)) * 100, 2)
        theo_above = round(100 - theo_below, 2)
    else:
        z_score = None
        theo_below = None
        theo_above = None

    return {
        "column": col,
        "threshold": threshold,
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "n": n,
        "z_score": z_score,
        "empirical": {
            "above": emp_above,
            "below": emp_below
        },
        "theoretical": {
            "above": theo_above,
            "below": theo_below
        },
        "conditional_placement": cond_placement,
        "interpretation": (
            f"P({col} > {threshold}) = {emp_above}% of students. "
            f"Among those, {cond_placement}% are placed."
        )
    }


# ─────────────────────────────────────────────
# ENDPOINT 4 — /api/probability/placement-prob
# P(Placed | CGPA range), P(Placed | IQ range)
# Grouped conditional probabilities
# ─────────────────────────────────────────────
@router.get("/probability/placement-prob")
def placement_probability():
    result = {}

    cgpa_bins = [0.0, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]
    cgpa_labels = ["F", "D", "C", "C+", "B", "B+", "A"]

    temp_df = df.copy()

    temp_df["cgpa_group"] = pd.cut(
        temp_df["CGPA"],
        bins=cgpa_bins,
        labels=cgpa_labels,
        right=False,
        include_lowest=True
    )

    cgpa_prob = (
        temp_df.groupby("cgpa_group", observed=True)["Placement"]
        .agg(["mean", "count"])
        .reset_index()
    )

    result["by_cgpa"] = [
        {
            "group": str(row["cgpa_group"]),
            "placement_prob": round(float(row["mean"]) * 100, 2),
            "count": int(row["count"]),
        }
        for _, row in cgpa_prob.iterrows()
    ]

    iq_bins = [0, 80, 90, 100, 110, 120, float("inf")]
    iq_labels = ["0-80", "80-90", "90-100", "100-110", "110-120", "120+"]

    temp_df["iq_group"] = pd.cut(
        temp_df["IQ"],
        bins=iq_bins,
        labels=iq_labels,
        right=False,
        include_lowest=True
    )

    iq_prob = (
        temp_df.groupby("iq_group", observed=True)["Placement"]
        .agg(["mean", "count"])
        .reset_index()
    )

    result["by_iq"] = [
        {
            "group": str(row["iq_group"]),
            "placement_prob": round(float(row["mean"]) * 100, 2),
            "count": int(row["count"]),
        }
        for _, row in iq_prob.iterrows()
    ]

    return result