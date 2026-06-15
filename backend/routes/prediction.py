import pandas as pd
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

router = APIRouter()

# ─────────────────────────────────────────────
# Load + Clean + Encode
# ─────────────────────────────────────────────
df_raw = pd.read_csv("data/dataSet.csv")
df = df_raw.copy()
df.dropna(inplace=True)
df = df[df["CGPA"].between(0, 10)]
df.reset_index(drop=True, inplace=True)
df["Placement"]             = df["Placement"].map({"Yes": 1, "No": 0})
df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})
df["CGPA"]                  = ((df["CGPA"] / 10) * 4).round(2)

# ─────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────
FEATURES = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Internship_Experience", "Extra_Curricular_Score",
    "Communication_Skills", "Projects_Completed",
]

X = df[FEATURES]
y = df["Placement"]

# Train/Test Split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# Scale features for Logistic Regression
# ─────────────────────────────────────────────
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Train Model — Logistic Regression
# ─────────────────────────────────────────────
lr_model  = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sc, y_train)
y_pred_lr = lr_model.predict(X_test_sc)
y_prob_lr = lr_model.predict_proba(X_test_sc)[:, 1]


# ─────────────────────────────────────────────
# ENDPOINT 1 — /api/model/metrics
# Logistic Regression evaluation metrics
# ─────────────────────────────────────────────
@router.get("/model/metrics")
def get_metrics():
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    return {
        "logistic_regression": {
            "accuracy":  round(accuracy_score(y_test, y_pred_lr) * 100, 2),
            "f1_score":  round(f1_score(y_test, y_pred_lr) * 100, 2),
            "auc":       round(roc_auc_score(y_test, y_prob_lr) * 100, 2),
            "confusion_matrix": {
                "tn": int(cm_lr[0][0]),
                "fp": int(cm_lr[0][1]),
                "fn": int(cm_lr[1][0]),
                "tp": int(cm_lr[1][1]),
            },
            "coefficients": {
                feat: round(float(coef), 4)
                for feat, coef in zip(FEATURES, lr_model.coef_[0])
            },
        },
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }


# ─────────────────────────────────────────────
# ENDPOINT 2 — /api/predict
# Predict placement for a student
# ─────────────────────────────────────────────
class StudentInput(BaseModel):
    IQ:                     float
    Prev_Sem_Result:        float
    CGPA:                   float
    Academic_Performance:   float
    Internship_Experience:  int    # 0 or 1
    Extra_Curricular_Score: float
    Communication_Skills:   float
    Projects_Completed:     int


@router.post("/predict")
def predict_placement(student: StudentInput):
    input_data = np.array([[
        student.IQ,
        student.Prev_Sem_Result,
        student.CGPA,
        student.Academic_Performance,
        student.Internship_Experience,
        student.Extra_Curricular_Score,
        student.Communication_Skills,
        student.Projects_Completed,
    ]])

    # Scale for logistic regression
    input_scaled = scaler.transform(input_data)

    # Logistic Regression prediction
    lr_pred = int(lr_model.predict(input_scaled)[0])
    lr_prob = round(float(lr_model.predict_proba(input_scaled)[0][1]) * 100, 2)

    verdict = "Placed" if lr_prob >= 50 else "Not Placed"

    # Top factors by absolute coefficient magnitude
    coef_importance = {
        feat: abs(float(coef))
        for feat, coef in zip(FEATURES, lr_model.coef_[0])
    }
    top_factors = sorted(coef_importance.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "verdict":     verdict,
        "probability": lr_prob,
        "logistic": {
            "prediction":  "Placed" if lr_pred == 1 else "Not Placed",
            "probability": lr_prob,
        },
        "top_factors": [
            {"factor": f, "importance": round(i * 100, 2)}
            for f, i in top_factors
        ],
        "message": (
            f"Based on the provided profile, this student has a "
            f"{lr_prob:.1f}% probability of getting placed."
        ),
    }