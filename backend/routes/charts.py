import pandas as pd
import numpy as np
from fastapi import APIRouter

router = APIRouter()

# ─────────────────────────────────────────────
# Load + Clean + Encode (same as stats.py)
# ─────────────────────────────────────────────
df_raw = pd.read_csv("data/dataSet.csv")

df = df_raw.copy()
df.dropna(inplace=True)
df = df[df["CGPA"].between(0, 10)]
df.reset_index(drop=True, inplace=True)

df["Placement"]            = df["Placement"].map({"Yes": 1, "No": 0})
df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})
df["CGPA"]                 = ((df["CGPA"] / 10) * 4).round(2)


# ─────────────────────────────────────────────
# CHART 1 — Placement Pie
# Placed vs Not Placed counts
# ─────────────────────────────────────────────
@router.get("/chart/placement-pie")
def placement_pie():
    placed     = int(df["Placement"].sum())
    not_placed = int((df["Placement"] == 0).sum())
    return [
        {"name": "Placed",     "value": placed},
        {"name": "Not Placed", "value": not_placed},
    ]


# ─────────────────────────────────────────────
# CHART 2 — CGPA Histogram
# Distribution of CGPA values in bins
# ─────────────────────────────────────────────
@router.get("/chart/cgpa-histogram")
def cgpa_histogram():
    counts, bin_edges = np.histogram(df["CGPA"], bins=15)
    result = []
    for i in range(len(counts)):
        label = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
        result.append({"range": label, "count": int(counts[i])})
    return result


# ─────────────────────────────────────────────
# CHART 3 — IQ Histogram
# Distribution of IQ values in bins
# ─────────────────────────────────────────────
@router.get("/chart/iq-histogram")
def iq_histogram():
    counts, bin_edges = np.histogram(df["IQ"], bins=15)
    result = []
    for i in range(len(counts)):
        label = f"{int(bin_edges[i])} - {int(bin_edges[i+1])}"
        result.append({"range": label, "count": int(counts[i])})
    return result


# ─────────────────────────────────────────────
# CHART 4 — CGPA Box Plot Data
# Q1, Median, Q3, Min, Max per Placement group
# ─────────────────────────────────────────────
@router.get("/chart/cgpa-boxplot")
def cgpa_boxplot():
    result = []
    labels = {0: "Not Placed", 1: "Placed"}
    for val, label in labels.items():
        group = df[df["Placement"] == val]["CGPA"]
        result.append({
            "group":  label,
            "min":    round(float(group.min()), 2),
            "q1":     round(float(group.quantile(0.25)), 2),
            "median": round(float(group.median()), 2),
            "q3":     round(float(group.quantile(0.75)), 2),
            "max":    round(float(group.max()), 2),
            "mean":   round(float(group.mean()), 2),
        })
    return result


# ─────────────────────────────────────────────
# CHART 5— Scatter: CGPA vs IQ
# Colored by Placement (sample 500 for performance)
# ─────────────────────────────────────────────
@router.get("/chart/scatter")
def scatter():
    sample = df[["CGPA", "IQ", "Placement"]].sample(
        n=min(500, len(df)), random_state=42
    )
    return sample.rename(columns={
        "CGPA":      "cgpa",
        "IQ":        "iq",
        "Placement": "placement",
    }).to_dict(orient="records")


# ─────────────────────────────────────────────
# CHART 6 — Internship Bar
# Placement rate with vs without internship
# ─────────────────────────────────────────────
@router.get("/chart/internship-bar")
def internship_bar():
    result = []
    labels = {0: "No Internship", 1: "Has Internship"}
    for val, label in labels.items():
        group        = df[df["Internship_Experience"] == val]
        total        = len(group)
        placed       = int(group["Placement"].sum())
        not_placed   = total - placed
        placement_rate = round((placed / total) * 100, 2) if total > 0 else 0
        result.append({
            "group":          label,
            "placed":         placed,
            "not_placed":     not_placed,
            "total":          total,
            "placement_rate": placement_rate,
        })
    return result
