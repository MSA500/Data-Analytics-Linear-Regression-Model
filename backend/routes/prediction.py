import pandas as pd
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error
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
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Train Models
# ─────────────────────────────────────────────

# Model 1 — Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sc, y_train)
y_pred_lr   = lr_model.predict(X_test_sc)
y_prob_lr   = lr_model.predict_proba(X_test_sc)[:, 1]

# Model 2 — Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf   = rf_model.predict(X_test)
y_prob_rf   = rf_model.predict_proba(X_test)[:, 1]

# Model 3 — Linear Regression (predict CGPA)
LIN_FEATURES = [
    "IQ", "Prev_Sem_Result", "Academic_Performance",
    "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed",
]
X_lin = df[LIN_FEATURES]
y_lin = df["CGPA"]
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)
lin_model = LinearRegression()
lin_model.fit(X_train_l, y_train_l)
y_pred_lin = lin_model.predict(X_test_l)


# ─────────────────────────────────────────────
# ENDPOINT 1 — /api/model/metrics
# All model evaluation metrics
# ─────────────────────────────────────────────
@router.get("/model/metrics")
def get_metrics():
    # Confusion matrix values
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # ROC curve data for Logistic Regression
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
    roc_data = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::10], tpr[::10])  # sample every 10th point
    ]

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
        "random_forest": {
            "accuracy":  round(accuracy_score(y_test, y_pred_rf) * 100, 2),
            "f1_score":  round(f1_score(y_test, y_pred_rf) * 100, 2),
            "auc":       round(roc_auc_score(y_test, y_prob_rf) * 100, 2),
            "confusion_matrix": {
                "tn": int(cm_rf[0][0]),
                "fp": int(cm_rf[0][1]),
                "fn": int(cm_rf[1][0]),
                "tp": int(cm_rf[1][1]),
            },
            "feature_importance": {
                feat: round(float(imp), 4)
                for feat, imp in zip(FEATURES, rf_model.feature_importances_)
            },
        },
        "linear_regression": {
            "r2":        round(float(lin_model.score(X_test_l, y_test_l)) * 100, 2),
            "mae":       round(float(mean_absolute_error(y_test_l, y_pred_lin)), 4),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_test_l, y_pred_lin))), 4),
            "coefficients": {
                feat: round(float(coef), 4)
                for feat, coef in zip(LIN_FEATURES, lin_model.coef_)
            },
            "intercept": round(float(lin_model.intercept_), 4),
            "equation":  "CGPA = " + " + ".join(
                [f"({round(float(c),4)} × {f})" for f, c in zip(LIN_FEATURES, lin_model.coef_)]
            ) + f" + {round(float(lin_model.intercept_), 4)}",
        },
        "roc_curve":     roc_data,
        "train_size":    len(X_train),
        "test_size":     len(X_test),
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
    lr_pred  = int(lr_model.predict(input_scaled)[0])
    lr_prob  = float(lr_model.predict_proba(input_scaled)[0][1]) * 100

    # Random Forest prediction
    rf_pred  = int(rf_model.predict(input_data)[0])
    rf_prob  = float(rf_model.predict_proba(input_data)[0][1]) * 100

    # Final verdict — average of both
    avg_prob = round((lr_prob + rf_prob) / 2, 2)
    verdict  = "Placed" if avg_prob >= 50 else "Not Placed"

    # Top factors (from RF feature importance)
    importance = dict(zip(FEATURES, rf_model.feature_importances_))
    top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "verdict":        verdict,
        "probability":    avg_prob,
        "logistic": {
            "prediction":  "Placed" if lr_pred == 1 else "Not Placed",
            "probability": round(lr_prob, 2),
        },
        "random_forest": {
            "prediction":  "Placed" if rf_pred == 1 else "Not Placed",
            "probability": round(rf_prob, 2),
        },
        "top_factors": [
            {"factor": f, "importance": round(float(i) * 100, 2)}
            for f, i in top_factors
        ],
        "message": (
            f"Based on the provided profile, this student has a "
            f"{avg_prob:.1f}% probability of getting placed."
        ),
    }
