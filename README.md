# 🎓 Student Placement Prediction System

A full-stack machine learning web application that predicts whether a student will get placed based on their academic and personal profile.

---

## 📌 Project Overview

This system uses **Logistic Regression** to analyze a student's profile and predict their placement probability. It provides real-time predictions through an interactive dashboard with model evaluation metrics and a live prediction tool.

---

## 🛠️ Tech Stack

| Layer       | Technology                        |
|-------------|-----------------------------------|
| Frontend    | Next.js, Tailwind CSS, Recharts   |
| Backend     | FastAPI (Python)                  |
| ML Models   | Scikit-learn (Logistic Regression)|
| Data        | Pandas, NumPy                     |

---

## 🤖 Machine Learning

### Model — Logistic Regression
- Trained on student academic data
- Features are **StandardScaler** normalized before training
- 80/20 train-test split with `random_state=42`

### Input Features
| Feature | Description |
|---|---|
| IQ | Student's IQ score |
| Prev Sem Result | Previous semester result (0–10) |
| CGPA | Cumulative GPA (0–4.0) |
| Academic Performance | Overall academic score (1–10) |
| Internship Experience | Has internship or not (0/1) |
| Extra Curricular Score | Co-curricular activities score (0–10) |
| Communication Skills | Communication rating (0–10) |
| Projects Completed | Number of projects done (0–5) |

### Evaluation Metrics
- **Accuracy** — Overall correct predictions
- **F1 Score** — Balance between precision and recall
- **AUC** — Model's ability to distinguish between placed/not placed
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN

---

## 🚀 Getting Started

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

App will be running at `http://localhost:3000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/model/metrics` | Returns model evaluation metrics |
| POST | `/api/predict` | Predicts placement for a student |

### Example — POST `/api/predict`
```json
{
  "IQ": 110,
  "Prev_Sem_Result": 8.0,
  "CGPA": 3.5,
  "Academic_Performance": 8,
  "Internship_Experience": 1,
  "Extra_Curricular_Score": 7,
  "Communication_Skills": 8,
  "Projects_Completed": 3
}
```

### Response
```json
{
  "verdict": "Placed",
  "probability": 84.32,
  "logistic": {
    "prediction": "Placed",
    "probability": 84.32
  },
  "top_factors": [
    { "factor": "CGPA", "importance": 134.0 },
    { "factor": "Communication_Skills", "importance": 91.0 },
    { "factor": "IQ", "importance": 82.0 }
  ],
  "message": "Based on the provided profile, this student has a 84.3% probability of getting placed."
}
```

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── routes/
│   │   └── ml_router.py      # ML endpoints & model training
│   └── data/
│       └── dataSet.csv       # Training dataset
│
└── frontend/
    ├── app/
    │   └── prediction/
    │       └── page.jsx      # Main prediction dashboard
    └── components/
```

---

## 📊 Dashboard Features

- **Model Performance Card** — Accuracy, F1, AUC at a glance
- **Confusion Matrix** — Visual breakdown of predictions
- **Feature Coefficients** — Which features impact placement most
- **Live Prediction Tool** — Sliders to input student profile and get instant prediction

---


Built as an academic ML project demonstrating end-to-end integration of machine learning with a modern web stack.
