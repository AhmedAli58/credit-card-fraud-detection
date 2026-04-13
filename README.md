## Problem

Credit card fraud costs financial institutions billions annually. The challenge is not building a classifier — it is building one that works under extreme class imbalance (0.17% fraud), minimizes false alarms that frustrate legitimate customers, and produces reliable risk scores for analyst prioritization.

---

## Approach

Three deliberate decisions drove the results:

- **scale_pos_weight over SMOTE** — XGBoost's native class weighting (578x) avoids synthetic noise while correctly penalizing missed fraud
- **F-beta threshold optimization** — threshold tuned to weight precision 2x over recall, matching real banking cost structure
- **Isotonic calibration** — ensures risk scores are statistically reliable, not just ranked

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9797 |
| PR-AUC | 0.8738 |
| Precision | 93% |
| Recall | 81% |
| False Positives | 7 |
| False Negatives | 19 |
| Business Cost | $9,570 |

Reviewing the top 1% of flagged transactions catches 90% of all fraud — reducing analyst workload by 99%.

---

## Stack

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=flat&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat&logo=pandas&logoColor=white)

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases, features V1-V28 PCA-transformed for confidentiality.

---

## Run Locally
```bash
git clone https://github.com/AhmedAli58/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```
