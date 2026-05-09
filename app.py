import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeleChurn AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Main title */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    color : white;
    # background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    # -webkit-background-clip: text;
    # -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.main-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}

/* Result cards */
.result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.25), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 2px;
    padding: 25px;
    text-align: center;
    animation: pulse-red 2s infinite;
}
.result-safe {
    background: linear-gradient(135deg, rgba(52,211,153,0.25), rgba(16,185,129,0.1));
    border: 2px solid rgba(52,211,153,0.4);
    border-radius: 20px;
    padding: 2rem;
    margin: 5rem 0rem 2rem;
    text-align: center;
    animation: pulse-green 3s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
    50% { box-shadow: 0 0 20px 6px rgba(239,68,68,0.15); }
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.3); }
    50% { box-shadow: 0 0 20px 6px rgba(52,211,153,0.15); }
}

.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }
.result-prob { font-size: 3rem; font-weight: 700; margin: 0.4rem 0; }
.result-prob-churn { color: #f87171; }
.result-prob-safe { color: #34d399; }
.result-desc { color: #94a3b8; font-size: 0.9rem; }

/* Metric boxes */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.metric-box .lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }

/* Section headers */
.section-header {
    color: #e2e8f0;
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Risk badge */
.risk-high { color: #f87171 !important; font-weight: 700; }
.risk-medium { color: #fbbf24 !important; font-weight: 700; }
.risk-low { color: #34d399 !important; font-weight: 700; }

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}

/* Selectbox, number input */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Slider */
.stSlider > div > div > div { background: rgba(167,139,250,0.3) !important; }

/* All text */
label, .stMarkdown p, .stText { color: #cbd5e1 !important; }
h1, h2, h3 { color: #f1f5f9 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* Tips box */
.tip-box {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}
.tip-box p { color: #c4b5fd !important; font-size: 0.85rem; margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_churn_data (1).csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return df

model, encoders = load_model()
raw_df = load_data()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title"> 📡 TeleChurn AI Predictor </h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Machine Learning–powered Customer Churn Intelligence Platform</p>', unsafe_allow_html=True)

# ─── Sidebar Inputs ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown('<div class="section-header">📞 Services</div>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown('<div class="section-header">💳 Billing</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), 10.0)
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

# ─── Encode Inputs ───────────────────────────────────────────────────────────────
def encode_input():
    raw = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'DeviceProtection': device_protection,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'PaymentMethod': payment_method,
        'PaperlessBilling': paperless_billing,
    }
    cat_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'TechSupport', 'DeviceProtection',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in cat_cols:
        le = encoders[col]
        try:
            raw[col] = le.transform([raw[col]])[0]
        except ValueError:
            raw[col] = 0

    order = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
             'MultipleLines', 'InternetService', 'OnlineSecurity', 'TechSupport',
             'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'Contract',
             'MonthlyCharges', 'TotalCharges', 'PaymentMethod', 'PaperlessBilling']
    return pd.DataFrame([raw])[order]

# ─── Charts ─────────────────────────────────────────────────────────────────────
def make_donut(churn_prob):
    safe_prob = 1 - churn_prob
    colors = ['#f87171', '#34d399'] if churn_prob > 0.5 else ['#34d399', '#f87171']
    labels = ['Churn Risk', 'Retention'] if churn_prob > 0.5 else ['Retention', 'Churn Risk']
    values = [churn_prob, safe_prob] if churn_prob > 0.5 else [safe_prob, churn_prob]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
        textinfo='label+percent',
        textfont=dict(color='white', size=13),
        pull=[0.05, 0],
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(
            text=f"<b>{churn_prob*100:.0f}%</b>",
            x=0.5, y=0.5, font_size=26,
            font_color='#f87171' if churn_prob > 0.5 else '#34d399',
            showarrow=False
        )]
    )
    return fig

def make_spider(input_df):
    features_display = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges / 200,
        'TotalCharges': min(total_charges / 8000, 1.0),
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Contract_risk': {'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.0}[contract],
        'Services': sum([
            phone_service == 'Yes',
            multiple_lines == 'Yes',
            internet_service != 'No',
            online_security == 'Yes',
            tech_support == 'Yes',
            device_protection == 'Yes',
            streaming_tv == 'Yes',
            streaming_movies == 'Yes',
        ]) / 8,
        'Digital': 1 if paperless_billing == 'Yes' else 0,
        'AutoPay': 1 if 'automatic' in payment_method.lower() else 0,
    }

    cats = list(features_display.keys())
    vals = [min(v / 72 if k == 'tenure' else v, 1.0) for k, v in features_display.items()]
    cats += [cats[0]]
    vals += [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill='toself',
        fillcolor='rgba(124,58,237,0.25)',
        line=dict(color='#a78bfa', width=2),
        marker=dict(color='#c4b5fd', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(255,255,255,0.03)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="#41689d", size=9),
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(color='#e2e8f0', size=11),
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.15)',
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
    )
    return fig

def make_feature_importance():
    features_labels = ['SeniorCitizen','Partner','Dependents','Tenure','PhoneService',
                       'MultipleLines','InternetService','OnlineSecurity','TechSupport',
                       'DeviceProtection','StreamingTV','StreamingMovies','Contract',
                       'MonthlyCharges','TotalCharges','PaymentMethod','PaperlessBilling']
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': features_labels, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=True).tail(10)

    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker=dict(
            color=df_imp['Importance'],
            colorscale=[[0, '#4c1d95'], [0.5, '#7c3aed'], [1, '#a78bfa']],
            line=dict(color='rgba(0,0,0,0)')
        ),
        text=[f"{v:.3f}" for v in df_imp['Importance']],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=10),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(tickfont=dict(color='#e2e8f0')),
        margin=dict(t=10, b=10, l=10, r=60),
        height=320,
    )
    return fig

def make_dataset_donut():
    counts = raw_df['Churn'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['Retained', 'Churned'],
        values=[counts.get('No', 0), counts.get('Yes', 0)],
        hole=0.6,
        marker=dict(colors=['#34d399', '#f87171'], line=dict(color='rgba(0,0,0,0)', width=0)),
        textinfo='label+percent',
        textfont=dict(color='white', size=12),
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        annotations=[dict(
            text="<b>Dataset</b>",
            x=0.5, y=0.5, font_size=13,
            font_color='#94a3b8',
            showarrow=False
        )]
    )
    return fig

def make_tenure_hist():
    churn_yes = raw_df[raw_df['Churn'] == 'Yes']['tenure']
    churn_no = raw_df[raw_df['Churn'] == 'No']['tenure']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=churn_no, name='Retained', marker_color='#34d399',
                               opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=churn_yes, name='Churned', marker_color='#f87171',
                               opacity=0.7, nbinsx=30))
    fig.update_layout(
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Tenure (months)', tickfont=dict(color='#94a3b8'),
                   gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#94a3b8')),
        yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=20, b=30, l=10, r=10),
        height=240,
    )
    return fig

# ─── Dataset Stats ────────────────────────────────────────────────────────────
churn_rate = (raw_df['Churn'] == 'Yes').mean()
avg_tenure = raw_df['tenure'].mean()
avg_monthly = raw_df['MonthlyCharges'].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'''<div class="metric-box"><div class="val">{len(raw_df):,}</div><div class="lbl">Total Customers</div></div>''', unsafe_allow_html=True)
with col2:
    st.markdown(f'''<div class="metric-box"><div class="val" style="color:#f87171">{churn_rate:.1%}</div><div class="lbl">Churn Rate</div></div>''', unsafe_allow_html=True)
with col3:
    st.markdown(f'''<div class="metric-box"><div class="val" style="color:#60a5fa">{avg_tenure:.0f}mo</div><div class="lbl">Avg Tenure</div></div>''', unsafe_allow_html=True)
with col4:
    st.markdown(f'''<div class="metric-box"><div class="val" style="color:#34d399">${avg_monthly:.0f}</div><div class="lbl">Avg Monthly $</div></div>''', unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────
if predict_btn:
    input_df = encode_input()
    proba = model.predict_proba(input_df)[0]
    churn_prob = proba[1]
    is_churn = churn_prob >= 0.5

    # Risk level
    if churn_prob >= 0.7:
        risk_level = "🔴 High Risk"
        risk_class = "risk-high"
    elif churn_prob >= 0.4:
        risk_level = "🟡 Medium Risk"
        risk_class = "risk-medium"
    else:
        risk_level = "🟢 Low Risk"
        risk_class = "risk-low"

    # Result + Donut
    left, right = st.columns([1, 1])
    with left:
        if is_churn:
            st.markdown(f'''
            <div class="result-churn">
                <div class="result-icon">⚠️</div>
                <div class="result-label">Likely to Churn</div>
                <div class="result-prob result-prob-churn">{churn_prob*100:.1f}%</div>
                <div class="result-desc">Churn probability · <span class="{risk_class}">{risk_level}</span></div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-label">Likely to Stay</div>
                <div class="result-prob result-prob-safe">{(1-churn_prob)*100:.1f}%</div>
                <div class="result-desc">Retention probability · <span class="{risk_class}">{risk_level}</span></div>
            </div>''', unsafe_allow_html=True)

        # Tips
        tips = []
        if contract == "Month-to-month":
            tips.append("📋 Offer long-term contract discounts")
        if internet_service == "Fiber optic" and online_security == "No":
            tips.append("🔒 Upsell online security bundle")
        if tenure < 12:
            tips.append("🎁 Early loyalty reward program")
        if monthly_charges > 80:
            tips.append("💰 Review pricing plan options")
        if payment_method == "Electronic check":
            tips.append("🏦 Incentivize automatic payment setup")

        if tips:
            st.markdown("**💡 Retention Recommendations**")
            for t in tips:
                st.markdown(f'<p>{t}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">🍩 Churn Probability</div>', unsafe_allow_html=True)
        st.plotly_chart(make_donut(churn_prob), use_container_width=True, config={'displayModeBar': False})

    # Spider + Feature Importance
    st.markdown("---")
    l2, r2 = st.columns([1, 1])
    with l2:
        st.markdown('<div class="section-header">🕸️ Customer Profile Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(make_spider(input_df), use_container_width=True, config={'displayModeBar': False})
    # with r2:
    #     st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)
    #     st.plotly_chart(make_feature_importance(), use_container_width=True, config={'displayModeBar': False})

else:
    # Default view — dataset analytics
    st.markdown("---")
    l, r = st.columns([1, 1.6])
    with l:
        st.markdown('<div class="section-header">🍩 Dataset Churn Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(make_dataset_donut(), use_container_width=True, config={'displayModeBar': False})
    with r:
        st.markdown('<div class="section-header">📈 Tenure vs Churn Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(make_tenure_hist(), use_container_width=True, config={'displayModeBar': False})

    # Contract bar
    st.markdown('<div class="section-header">📋 Churn Rate by Contract Type</div>', unsafe_allow_html=True)
    contract_churn = raw_df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    contract_churn.columns = ['Contract', 'Churn Rate']
    fig_c = go.Figure(go.Bar(
        x=contract_churn['Contract'],
        y=contract_churn['Churn Rate'],
        marker=dict(
            color=contract_churn['Churn Rate'],
            colorscale=[[0, '#34d399'], [0.5, '#fbbf24'], [1, '#f87171']],
        ),
        text=[f"{v:.1%}" for v in contract_churn['Churn Rate']],
        textposition='outside',
        textfont=dict(color='white'),
    ))
    fig_c.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#e2e8f0'), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(255,255,255,0.08)',
                   tickformat='.0%'),
        margin=dict(t=20, b=10, l=10, r=10),
        height=280,
    )
    st.plotly_chart(fig_c, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class="glass-card" style="text-align:center; color:#94a3b8;">
        ← Fill in the customer details on the left sidebar and click <b style="color:#a78bfa">Predict Churn Risk</b> to get AI predictions
    </div>
    """, unsafe_allow_html=True)