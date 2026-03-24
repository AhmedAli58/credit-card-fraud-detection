# Credit Card Fraud Detection System

Financial institutions lose billions annually to credit card fraud. This project builds a production-grade fraud detection system that balances two competing priorities — catching fraud while minimizing disruption to legitimate customers.

**Live Dashboard:** [View Here](https://credit-card-fraud-detection-33azof3lzauheeppgmaxzx.streamlit.app) | **Notebook:** [Kaggle](https://www.kaggle.com/code/ahmedallii/credit-card-fraud-detection-system)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat&logo=numpy&logoColor=white)

| Tool | Purpose |
|------|---------|
| XGBoost | Core fraud classifier with scale_pos_weight for imbalance |
| Scikit-learn | SMOTE, calibration, threshold tuning, evaluation metrics |
| Streamlit | Interactive dashboard deployment |
| Plotly | Risk score distribution, Sankey, PR curve, scatter plots |
| Pandas / NumPy | Data manipulation and feature engineering |

---

## The Problem

Standard fraud detection models optimize for accuracy — misleading when fraud is only 0.17% of transactions. A model that flags everything as legitimate achieves 99.83% accuracy while catching zero fraud.

The real challenge is building a system that:
- Catches as much fraud as possible (recall)
- Avoids blocking legitimate customers (precision)
- Provides calibrated risk scores for analyst prioritization

---

## Key Design Decisions

**1. scale_pos_weight instead of SMOTE**
Used XGBoost's native `scale_pos_weight=578` rather than oversampling — penalizes missing fraud 578x more, matching the real cost ratio without introducing synthetic noise.

**2. F-beta threshold optimization**
Optimized threshold using F-beta (β=0.5) which weights precision twice as much as recall — the right tradeoff for a bank that prioritizes customer experience.

**3. Isotonic calibration**
Raw XGBoost probabilities are overconfident. Isotonic regression ensures a score of 0.8 means 80% fraud probability — critical for the risk ranking system.

---

## Business Impact

| Metric | Value |
|--------|-------|
| Transactions Analyzed | 56,962 |
| Fraud Detected | 79 / 98 cases |
| False Alarms | 7 legitimate customers blocked |
| Precision | 93% |
| Recall | 81% |
| ROC-AUC | 0.9797 |
| PR-AUC | 0.8738 |
| Business Cost | $9,570 |

> Reviewing just the **top 1% of highest-risk transactions catches 90% of all fraud** — analysts review 570 transactions instead of 56,962.

---

## Model Performance
```
Confusion Matrix

                 Predicted: Legit    Predicted: Fraud
Actual: Legit       56,857               7        ← 7 false alarms
Actual: Fraud          19              79        ← 79 caught, 19 missed

Threshold: 0.884 | Optimized for banking precision
```

---

## Dashboard

Built for three audiences:

- **Executives** — KPI cards: detection rate, precision, business cost
- **Data Scientists** — PR curve, ROC-AUC, Sankey flow, calibration metrics  
- **Fraud Analysts** — Alert queue of 82 high-risk transactions sorted by risk score

---

## Pipeline
```
Raw Transactions (284,807)
        ↓
Feature Scaling (StandardScaler — Amount + Time)
        ↓
XGBoost (scale_pos_weight=578, n_estimators=300)
        ↓
F-beta Threshold Optimization (β=0.5 → threshold=0.884)
        ↓
Isotonic Calibration
        ↓
Risk Scores + Binary Decisions
        ↓
Streamlit Dashboard
```

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 European cardholder transactions. Features V1-V28 are PCA-transformed for confidentiality.

---

## Run Locally
```bash
git clone https://github.com/AhmedAli58/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```