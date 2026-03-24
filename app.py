import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

st.set_page_config(
    page_title="Fraud Intelligence | Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #070d1a; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

.top-bar {
    background: #0b1222;
    border-bottom: 1px solid #1a2540;
    padding: 12px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-bar-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #e2e8f0;
    text-transform: uppercase;
}
.top-bar-meta { font-size: 0.72rem; color: #4b5563; letter-spacing: 0.5px; }
.top-bar-meta span { margin-left: 16px; }
.top-bar-meta .hl { color: #00d4aa; }
.top-bar-meta .wn { color: #ffaa00; }

.main-content { padding: 20px 28px; }

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    margin-top: 4px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.kpi-card {
    background: #0d1526;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.blue::before { background: #3b82f6; }
.kpi-card.red::before { background: #ef4444; }
.kpi-card.green::before { background: #00d4aa; }
.kpi-card.orange::before { background: #f59e0b; }
.kpi-card.purple::before { background: #8b5cf6; }

.kpi-value { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; line-height: 1; margin-bottom: 6px; }
.kpi-label { font-size: 0.65rem; font-weight: 500; color: #4b5563; text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-sub { font-size: 0.65rem; color: #374151; margin-top: 4px; }

.blue-val { color: #3b82f6; }
.red-val { color: #ef4444; }
.green-val { color: #00d4aa; }
.orange-val { color: #f59e0b; }
.purple-val { color: #8b5cf6; }

.chart-card {
    background: #0d1526;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 16px;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_excel('data/fraud_dashboard_v2.xlsx')

df = load_data()

total = len(df)
actual_fraud = int(df['Actual'].sum())
tp = int(((df['Actual']==1) & (df['Predicted']==1)).sum())
fp = int(((df['Actual']==0) & (df['Predicted']==1)).sum())
fn = int(((df['Actual']==1) & (df['Predicted']==0)).sum())
tn = int(((df['Actual']==0) & (df['Predicted']==0)).sum())
precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
business_cost = (fp*10)+(fn*500)

PLOT = dict(
    plot_bgcolor='#0d1526',
    paper_bgcolor='#0d1526',
    font=dict(color='#6b7280', family='Inter', size=11),
    margin=dict(l=45, r=20, t=35, b=45),
    hoverlabel=dict(bgcolor='#111827', bordercolor='#1a2540', font_color='#e2e8f0'),
    legend=dict(bgcolor='#0d1526', bordercolor='#1a2540', borderwidth=1, font=dict(color='#9ca3af'))
)

def ax(extra=None):
    base = dict(gridcolor='#1a2540', zerolinecolor='#1a2540',
                linecolor='#1a2540', tickfont=dict(color='#4b5563'))
    if extra:
        base.update(extra)
    return base

# Header
st.markdown("""
<div class="top-bar">
    <div>
        <span style="color:#3b82f6;font-size:1rem;margin-right:10px">⬡</span>
        <span class="top-bar-title">Fraud Intelligence Dashboard</span>
        <span style="color:#1a2540;margin:0 12px">|</span>
        <span style="font-size:0.72rem;color:#374151">Credit Card Risk Monitoring System</span>
    </div>
    <div class="top-bar-meta">
        <span>Model: <span class="hl">XGBoost + Isotonic</span></span>
        <span>Threshold: <span class="wn">0.786</span></span>
        <span>ROC-AUC: <span class="hl">0.9789</span></span>
        <span>PR-AUC: <span class="hl">0.8706</span></span>
        <span>Test Set: <span class="hl">56,962 txns</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# KPI Row 1
st.markdown('<div class="section-label">Transaction Overview</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card blue">
        <div class="kpi-value blue-val">{total:,}</div>
        <div class="kpi-label">Total Transactions</div>
        <div class="kpi-sub">Test set evaluation</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-value red-val">{actual_fraud}</div>
        <div class="kpi-label">Confirmed Fraud</div>
        <div class="kpi-sub">0.17% fraud rate</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-value green-val">{tp}</div>
        <div class="kpi-label">Fraud Detected</div>
        <div class="kpi-sub">{recall:.0%} detection rate</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-value orange-val">{fn}</div>
        <div class="kpi-label">Fraud Missed</div>
        <div class="kpi-sub">False negatives</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-value orange-val">{fp}</div>
        <div class="kpi-label">False Alarms</div>
        <div class="kpi-sub">Legit flagged as fraud</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-value red-val">${business_cost:,}</div>
        <div class="kpi-label">Business Cost</div>
        <div class="kpi-sub">FP=$10 · FN=$500</div>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Row 2
st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card green">
        <div class="kpi-value green-val">{precision:.1%}</div>
        <div class="kpi-label">Precision</div>
        <div class="kpi-sub">Of flagged, % real fraud</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-value green-val">{recall:.1%}</div>
        <div class="kpi-label">Recall</div>
        <div class="kpi-sub">Of all fraud, % caught</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-value green-val">{f1:.1%}</div>
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-sub">Harmonic mean</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-value blue-val">97.89%</div>
        <div class="kpi-label">ROC-AUC</div>
        <div class="kpi-sub">Discrimination ability</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-value blue-val">87.06%</div>
        <div class="kpi-label">PR-AUC</div>
        <div class="kpi-sub">Precision-recall tradeoff</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-value purple-val">86.50%</div>
        <div class="kpi-label">Calibrated PR-AUC</div>
        <div class="kpi-sub">Post-calibration score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Charts Row 1
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Risk Score Separation — Fraud vs Legitimate</div>', unsafe_allow_html=True)

    # KDE-style density using histogram with smoothing
    fraud_scores = df[df['Actual']==1]['RiskScore'].values
    legit_scores = df[df['Actual']==0]['RiskScore'].values

    fig1 = go.Figure()

    # Log scale bins for better separation
    bins = np.logspace(-6, 0, 80)

    fig1.add_trace(go.Histogram(
        x=legit_scores, name='Legitimate',
        xbins=dict(start=0, end=1, size=0.01),
        marker=dict(color='#3b82f6', opacity=0.6),
        histnorm='probability density'
    ))
    fig1.add_trace(go.Histogram(
        x=fraud_scores, name='Fraud',
        xbins=dict(start=0, end=1, size=0.01),
        marker=dict(color='#ef4444', opacity=0.85),
        histnorm='probability density'
    ))
    fig1.add_vrect(x0=0.786, x1=1.0,
        fillcolor='rgba(239,68,68,0.05)',
        line_width=0,
        annotation_text='Alert Zone',
        annotation_font_color='#ef4444',
        annotation_font_size=9
    )
    fig1.add_vline(x=0.786, line_dash='dot',
        line_color='#f59e0b', line_width=1.5,
        annotation_text='Threshold 0.786',
        annotation_font_color='#f59e0b',
        annotation_font_size=9
    )
    fig1.update_layout(**PLOT, barmode='overlay', height=300,
        xaxis=ax(dict(title='Risk Score', range=[0,1])),
        yaxis=ax(dict(title='Density')),
        title=dict(
            text='Clear separation: fraud scores cluster near 1.0, legitimate near 0',
            font=dict(size=10, color='#374151'), x=0
        )
    )
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Transaction Flow — Sankey</div>', unsafe_allow_html=True)

    fig2 = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15, thickness=20,
            line=dict(color='#1a2540', width=0.5),
            label=['All Transactions', 'Flagged (86)', 'Not Flagged (56,876)',
                   'True Fraud (79)', 'False Alarm (7)', 'Missed Fraud (19)', 'Safe (56,857)'],
            color=['#3b82f6', '#f59e0b', '#374151',
                   '#00d4aa', '#f59e0b', '#ef4444', '#1e3a5f'],
            x=[0.01, 0.45, 0.45, 0.99, 0.99, 0.99, 0.99],
            y=[0.5, 0.2, 0.75, 0.05, 0.35, 0.65, 0.9]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 5, 6],
            value=[86, 56876, 79, 7, 19, 56857],
            color=['rgba(245,158,11,0.3)', 'rgba(55,65,81,0.2)',
                   'rgba(0,212,170,0.3)', 'rgba(245,158,11,0.3)',
                   'rgba(239,68,68,0.3)', 'rgba(30,58,95,0.2)']
        )
    ))
    fig2.update_layout(**PLOT, height=300,
        title=dict(
            text='92.7% of flagged transactions are confirmed fraud',
            font=dict(size=10, color='#374151'), x=0
        )
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# Charts Row 2
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Precision-Recall Curve</div>', unsafe_allow_html=True)

    # Simulate PR curve from risk scores
    pr_data = pd.DataFrame({
        'actual': df['Actual'].values,
        'score': df['RiskScore'].values
    })
    thresholds_pr = np.linspace(0, 1, 200)
    precisions, recalls = [], []
    for t in thresholds_pr:
        pred = (pr_data['score'] >= t).astype(int)
        tp_t = ((pr_data['actual']==1) & (pred==1)).sum()
        fp_t = ((pr_data['actual']==0) & (pred==1)).sum()
        fn_t = ((pr_data['actual']==1) & (pred==0)).sum()
        p = tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 1
        r = tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0
        precisions.append(p)
        recalls.append(r)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=recalls, y=precisions,
        mode='lines', name='PR Curve',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'
    ))
    fig3.add_trace(go.Scatter(
        x=[recall], y=[precision],
        mode='markers', name='Operating Point',
        marker=dict(color='#f59e0b', size=10, symbol='diamond',
                    line=dict(color='white', width=1))
    ))
    fig3.update_layout(**PLOT, height=280,
        xaxis=ax(dict(title='Recall', range=[0,1])),
        yaxis=ax(dict(title='Precision', range=[0,1])),
        title=dict(text=f'PR-AUC = 0.8706 · Operating point marked',
                   font=dict(size=10, color='#374151'), x=0)
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cumulative Fraud Caught</div>', unsafe_allow_html=True)

    sorted_df = df.sort_values('RiskScore', ascending=False).reset_index(drop=True)
    sorted_df['cumulative_fraud'] = sorted_df['Actual'].cumsum()
    sorted_df['pct_reviewed'] = (sorted_df.index + 1) / len(sorted_df) * 100
    sorted_df['pct_fraud_caught'] = sorted_df['cumulative_fraud'] / actual_fraud * 100

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=sorted_df['pct_reviewed'],
        y=sorted_df['pct_fraud_caught'],
        mode='lines', name='Model',
        line=dict(color='#00d4aa', width=2.5)
    ))
    fig4.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode='lines', name='Random',
        line=dict(color='#374151', width=1, dash='dash')
    ))
    fig4.add_vline(x=1, line_dash='dot', line_color='#f59e0b',
        annotation_text='Top 1%', annotation_font_color='#f59e0b',
        annotation_font_size=9)
    top1_caught = sorted_df[sorted_df['pct_reviewed'] <= 1]['cumulative_fraud'].max()
    top1_pct = top1_caught / actual_fraud * 100
    fig4.add_annotation(
        x=1, y=top1_pct,
        text=f'{top1_pct:.0f}% fraud<br>caught',
        font=dict(color='#f59e0b', size=9),
        showarrow=True, arrowcolor='#f59e0b',
        arrowsize=0.8, ax=30, ay=-20
    )
    fig4.update_layout(**PLOT, height=280,
        xaxis=ax(dict(title='% Transactions Reviewed', range=[0,20])),
        yaxis=ax(dict(title='% Fraud Caught', range=[0,100])),
        title=dict(text='Reviewing top 1% catches ~90% of all fraud',
                   font=dict(size=10, color='#374151'), x=0)
    )
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Amount vs Risk Score</div>', unsafe_allow_html=True)

    sample = df.sample(min(3000, len(df)), random_state=42)
    fraud_sample = df[df['Actual']==1]

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=sample[sample['Actual']==0]['RiskScore'],
        y=sample[sample['Actual']==0]['AmountUSD'],
        mode='markers', name='Legitimate',
        marker=dict(color='#3b82f6', size=3, opacity=0.4)
    ))
    fig5.add_trace(go.Scatter(
        x=fraud_sample['RiskScore'],
        y=fraud_sample['AmountUSD'],
        mode='markers', name='Fraud',
        marker=dict(color='#ef4444', size=7, opacity=0.9,
                    line=dict(color='white', width=0.5))
    ))
    fig5.update_layout(**PLOT, height=280,
        xaxis=ax(dict(title='Risk Score')),
        yaxis=ax(dict(title='Amount (Scaled)')),
        title=dict(text='High-risk fraud transactions visible top-right',
                   font=dict(size=10, color='#374151'), x=0)
    )
    st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# Alert Queue
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Alert Queue — High Risk Transactions Requiring Review</div>', unsafe_allow_html=True)

high_risk = df[df['RiskScore'] > 0.5][
    ['AmountUSD','RiskScore','IsFraud','IsPredictedFraud','RiskBucket']
].sort_values('RiskScore', ascending=False).copy()

high_risk['Status'] = high_risk.apply(
    lambda r: '🔴 Confirmed Fraud' if r['IsFraud']=='Fraud' and r['IsPredictedFraud']=='Fraud'
    else ('⚠️ False Alarm' if r['IsFraud']=='Legitimate'
    else '🔴 Missed Fraud'), axis=1
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    confirmed = len(high_risk[high_risk['IsFraud']=='Fraud'])
    st.markdown(f"""
    <div style="background:#0a1a0f;border:1px solid #1a3a2a;border-radius:6px;
    padding:10px 14px;margin-bottom:12px">
        <span style="font-size:1.1rem;font-weight:700;color:#00d4aa">{confirmed}</span>
        <span style="font-size:0.68rem;color:#4b5563;margin-left:8px;
        text-transform:uppercase;letter-spacing:1px">Confirmed Fraud in Queue</span>
    </div>""", unsafe_allow_html=True)
with col_b:
    false_alarms = len(high_risk[high_risk['IsFraud']=='Legitimate'])
    st.markdown(f"""
    <div style="background:#1a0f0a;border:1px solid #3a1a0a;border-radius:6px;
    padding:10px 14px;margin-bottom:12px">
        <span style="font-size:1.1rem;font-weight:700;color:#f59e0b">{false_alarms}</span>
        <span style="font-size:0.68rem;color:#4b5563;margin-left:8px;
        text-transform:uppercase;letter-spacing:1px">False Alarms to Clear</span>
    </div>""", unsafe_allow_html=True)
with col_c:
    q_prec = confirmed/len(high_risk) if len(high_risk)>0 else 0
    st.markdown(f"""
    <div style="background:#0a0f1a;border:1px solid #1a2540;border-radius:6px;
    padding:10px 14px;margin-bottom:12px">
        <span style="font-size:1.1rem;font-weight:700;color:#3b82f6">{q_prec:.1%}</span>
        <span style="font-size:0.68rem;color:#4b5563;margin-left:8px;
        text-transform:uppercase;letter-spacing:1px">Queue Precision</span>
    </div>""", unsafe_allow_html=True)

display = high_risk[['AmountUSD','RiskScore','Status','RiskBucket']].copy()
display.columns = ['Amount (USD)','Risk Score','Status','Risk Bucket']
display['Amount (USD)'] = display['Amount (USD)'].round(2)
display['Risk Score'] = display['Risk Score'].round(4)

st.dataframe(
    display,
    use_container_width=True,
    height=300,
    column_config={
        'Risk Score': st.column_config.ProgressColumn(
            'Risk Score', min_value=0, max_value=1, format='%.4f'
        ),
        'Amount (USD)': st.column_config.NumberColumn('Amount (USD)', format='$%.2f'),
    }
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:10px;border-top:1px solid #1a2540;
font-size:0.65rem;color:#374151;letter-spacing:1px">
FRAUD INTELLIGENCE SYSTEM · XGBOOST + ISOTONIC CALIBRATION · THRESHOLD 0.786 · ROC-AUC 0.9789
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)