# Credit Card Fraud Detection Dashboard

Real-time fraud risk monitoring dashboard built with Streamlit and XGBoost, deployed on Streamlit Cloud.

## Live Demo
[View Dashboard](https://credit-card-fraud-detection-33azof3lzauheeppgmaxzx.streamlit.app)

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9797 |
| PR-AUC | 0.8738 |
| Precision | 93.0% |
| Recall | 81.0% |
| F1 Score | 86.6% |
| Business Cost | $9,570 |
| False Positives | 7 |
| False Negatives | 19 |

---

## Dashboard Sections

- Executive KPIs — transaction overview and model performance
- Risk Score Distribution — fraud vs legitimate separation
- Transaction Flow Sankey — visual breakdown of model decisions
- Precision-Recall Curve — model quality at every threshold
- Cumulative Fraud Caught — business value of risk ranking
- Amount vs Risk Score — transaction pattern analysis
- Alert Queue — high risk transactions requiring review

---

## Model Details

- Algorithm: XGBoost with scale_pos_weight=578 for class imbalance
- Threshold: F-beta optimized at 0.884 for banking precision
- Calibration: Isotonic regression for reliable probabilities
- Dataset: 284,807 transactions, 0.17% fraud rate

---

## Tech Stack

- Python 3.12, XGBoost, Scikit-learn
- Streamlit 1.43, Plotly 5.24
- Pandas, NumPy

---

## Dataset

Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Kaggle Notebook

Full modeling pipeline:
https://www.kaggle.com/code/ahmedallii/creditcard

---

## How to Run
```bash
git clone https://github.com/AhmedAli58/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```