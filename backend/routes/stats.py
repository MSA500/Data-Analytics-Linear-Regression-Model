import pandas as pd
import numpy as np
from fastapi import APIRouter
from scipy import stats       

router = APIRouter()

df_raw = pd.read_csv("data/dataSet.csv") #LOading data file 
#Data can be in raw form so we process the data in first step by removing nulls or invalid fields
#by analyzing data(done in seperate file) we came to know that there is no null field
#but CGPA goes out of bound

df = df_raw.copy()
df.dropna(inplace=True)          # drop rows with nulls if any
df = df[df["CGPA"].between(0,10)] # CGPA should between 0,10
df.reset_index(drop=True, inplace=True)

#Convert strings in to numaric form
# Placement: Yes = 1, No = 0
df["Placement"] = df["Placement"].map({"Yes": 1, "No": 0})

# Internship_Experience: Yes = 1, No = 0
df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})

#currently CGPA is in scale of 10 lets convert it in to scale of 4.0 as most universities have that scale
df["CGPA"] = (df["CGPA"]/10)*4


def get_data_info():
    info = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "placed_count": int(df["Placement"].sum()),
        "not_placed_count": int((df["Placement"] == 0).sum()),
        "placement_rate": round(df["Placement"].mean() * 100, 2),
        "internship_yes": int(df["Internship_Experience"].sum()),
        "internship_no": int((df["Internship_Experience"] == 0).sum()),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(df_raw[col].isnull().sum()) for col in df_raw.columns},
        "value_counts": {
            "Placement": df["Placement"].value_counts().to_dict(),
            "Internship_Experience": df["Internship_Experience"].value_counts().to_dict(),
        },
    }
    return info


# ─────────────────────────────────────────────
# STEP 5 — API Endpoints
# ─────────────────────────────────────────────

@router.get("/data")
def get_data(page: int = 1, limit: int = 10000, placement: int = 3):
    """
    Returns paginated dataset rows.
    Query params:
      - page: page number (default 1)
      - limit: rows per page (default 100)
      - placement: filter by 'All', 'Yes', or 'No'
    """
    filtered = df.copy()

    # Filter by placement if requested
    if placement in [0, 1]:
        filtered = filtered[filtered["Placement"] == placement]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    page_data = filtered.iloc[start:end]

    # Replace NaN with None for safe JSON serialization
    records = page_data[
        [
            "College_ID", "IQ", "Prev_Sem_Result", "CGPA",
            "Academic_Performance", "Internship_Experience",
            "Extra_Curricular_Score", "Communication_Skills",
            "Projects_Completed", "Placement"
        ]
    ].replace({np.nan: None}).to_dict(orient="records")

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": int(np.ceil(total / limit)),
        "data": records,
    }


@router.get("/info")
def get_info():
    """
    Returns dataset meta info:
    shape, column names, dtypes, null counts, value counts, placement stats.
    """
    return get_data_info()


@router.get("/summary")
def get_summary():
    """
    Returns describe() stats for all numeric columns.
    """
    numeric_cols = [
        "IQ", "Prev_Sem_Result", "CGPA",
            "Academic_Performance", "Internship_Experience",
            "Extra_Curricular_Score", "Communication_Skills",
            "Projects_Completed", "Placement"
    ]
    desc = df[numeric_cols].describe().round(4)

    result = {}
    for col in numeric_cols:
        result[col] = {stat: float(desc.loc[stat, col]) for stat in desc.index}

    return result


@router.get("/test")
def get_test():
    return "Working... I am stats.py"

@router.get("/stats")
def get_stats():
    numeric_cols = [
        "IQ", "Prev_Sem_Result", "CGPA",
        "Academic_Performance", "Extra_Curricular_Score",
        "Communication_Skills", "Projects_Completed"
    ]

    result = {}
    for col in numeric_cols:
        data = df[col].dropna()
        n = len(data)

        mean    = float(data.mean())
        median  = float(data.median())
        mode    = float(data.mode()[0])
        std     = float(data.std())
        var     = float(data.var())
        minimum = float(data.min())
        maximum = float(data.max())
        q1      = float(data.quantile(0.25))
        q3      = float(data.quantile(0.75))
        iqr     = round(q3 - q1, 4)
        skew    = float(round(data.skew(), 4))

        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=mean,
            scale=stats.sem(data)
        )

        result[col] = {
            "mean":     round(mean, 4),
            "median":   round(median, 4),
            "mode":     round(mode, 4),
            "std":      round(std, 4),
            "variance": round(var, 4),
            "min":      round(minimum, 4),
            "max":      round(maximum, 4),
            "q1":       round(q1, 4),
            "q3":       round(q3, 4),
            "iqr":      round(iqr, 4),
            "skewness": round(skew, 4),
            "count":    n,
            "ci_95": {
                "lower": round(float(ci_low), 4),
                "upper": round(float(ci_high), 4),
            },
        }

    # Grouped by Placement (0 = Not Placed, 1 = Placed)
    grouped = {}
    labels = {0: "Not Placed", 1: "Placed"}
    for val, label in labels.items():
        group = df[df["Placement"] == val]
        grouped[label] = {}
        for col in numeric_cols:
            grouped[label][col] = {
                "mean":   round(float(group[col].mean()), 4),
                "median": round(float(group[col].median()), 4),
                "std":    round(float(group[col].std()), 4),
            }

    return {
        "overall": result,
        "grouped": grouped,
    }