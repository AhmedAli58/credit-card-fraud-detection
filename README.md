# Credit Card Fraud Detection

Detecting fraud in credit card transactions is challenging due to extreme class imbalance (only 0.17% of transactions are fraudulent). The objective is to identify most fraud cases while minimizing false alarms and reducing manual review effort.

---

## Problem

Financial institutions need systems that:
- Detect fraudulent transactions accurately  
- Avoid flagging legitimate users unnecessarily  
- Reduce the volume of transactions requiring manual review  

---

## Approach

- **Data Split**  
  Stratified 80/20 train-test split  
  5-fold cross-validation used for model tuning, with final metrics evaluated on a held-out test set  

- **Handling Class Imbalance**  
  SMOTE applied within training folds via pipeline to avoid data leakage  

- **Model**  
  XGBoost classifier trained on transaction features  

- **Threshold Optimization**  
  Decision threshold tuned using F-beta (β=0.5) to emphasize precision and reduce false alarms  

- **Calibration**  
  Isotonic calibration applied to ensure predicted probabilities reflect true likelihood of fraud  

---

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9797 |
| PR-AUC | 0.8738 |
| Precision | 93% |
| Recall | 81% |

---

## Business Impact

Transactions are ranked by predicted risk score.  
Reviewing the **top 1% highest-risk transactions captures ~90% of fraud**, reducing manual review workload by ~99%.

---

## Dataset

- 284,807 transactions  
- 492 fraud cases (0.17%)  
- Source: Kaggle Credit Card Fraud Detection dataset  

---

## Tools

- Python  
- XGBoost  
- Scikit-learn  
- Pandas  
- Streamlit  

---

## Run Locally

```bash
git clone https://github.com/AhmedAli58/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
