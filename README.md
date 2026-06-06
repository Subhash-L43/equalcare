---
title: EqualCare
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# 🏥 EqualCare — Hospital District Recommender

> Find the most **affordable** and **quality** hospital district for your health needs across India.

**SDG 1 — No Poverty | PES University — Section F**

---

## What It Does

EqualCare helps patients find the **Top 5 best hospital districts** in any Indian state for a specific disease or health need.

**User inputs:**
- Select a **State** (25 states covered)
- Select a **Disease / Health Need** (10 categories)

**App outputs:**
- Top 5 ranked districts with scores
- Hospital Rating, Hospital Score, Affordability Score
- ML-predicted Recommendation label per district

---

## ML Model

| Detail | Info |
|---|---|
| Model | GradientBoostingClassifier |
| Features | Hospital_Rating, District_Avg_Rating, Hospital_Score, Affordability_Score |
| Target | Recommendation (Highly Recommended / Recommended / Moderate / Not Recommended) |
| Dataset | 2,566 hospitals across 204 districts in 25 Indian states |
| Train/Test Split | 80% / 20% |

---

## Disease Categories

- Basic Primary Care
- Cancer Treatment
- Cardiac Care
- Diabetes Management
- Eye Care
- General Medicine
- Kidney Disease
- Maternity & Child Health
- Neurology
- Orthopaedics

---

## Project Structure

```
├── app.py                  # Streamlit frontend
├── requirements.txt        # Python dependencies
├── model.pkl               # Trained GradientBoosting model
├── encoder.pkl             # LabelEncoder for Recommendation column
├── hospital_data.pkl       # Cleaned hospital dataset
├── meta.pkl                # States, diseases, accuracy metadata
└── README.md               # This file
```

---

## How to Run Locally

```bash
# 1. Clone or download the repo
# 2. Create a virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> ⚠️ Python 3.14 is NOT supported. Use Python 3.11.

---

## How Ranking Works

For each district the app calculates a **Combined Score**:

```
Combined Score = (
    Avg Hospital Rating +
    Avg District Avg Rating +
    Avg Hospital Score +
    Avg Affordability Score +
    Avg ML Label Score × 2
) ÷ 6
```

Districts are ranked highest → lowest by Combined Score. Top 5 are shown.

---

## Built With

- Python 3.11
- Streamlit
- scikit-learn
- pandas
- numpy

---

*EqualCare — Making quality healthcare accessible to all.*
