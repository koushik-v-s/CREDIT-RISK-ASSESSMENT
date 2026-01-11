import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from credit_model import CreditRiskModel
from metrics import ks_statistic, gini_coefficient, risk_bucket
from stress import apply_stress
from portfolio import portfolio_metrics
from validation import population_stability_index
from decision import credit_decision
from explainability import logistic_explain
from capital import approximate_rwa

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Enterprise Credit Risk Platform",
    layout="wide"
)

st.title("🏦 Enterprise Credit Risk Platform")
st.caption("Wholesale Credit Risk | Quantitative Risk | Banking Analyst")

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------
st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["logistic", "random_forest"]
)

lgd = st.sidebar.slider(
    "LGD (Loss Given Default)",
    0.0, 1.0, 0.45
)

stress_level = st.sidebar.selectbox(
    "Stress Scenario",
    ["None", "Mild", "Severe"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("data/loan_data.csv")
feature_cols = df.drop("default", axis=1).columns.tolist()

# --------------------------------------------------
# APPLY STRESS SCENARIO
# --------------------------------------------------
if stress_level != "None":
    df = apply_stress(df, stress_level)

# --------------------------------------------------
# TRAIN CREDIT RISK MODEL
# --------------------------------------------------
crm = CreditRiskModel(model_type)

X_train, X_test, y_train, y_test = crm.split_data(df)
crm.fit(X_train, y_train)

# Convert arrays back to DataFrames for alignment
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
X_test_df = pd.DataFrame(X_test, columns=feature_cols)

# --------------------------------------------------
# PREDICT PDs
# --------------------------------------------------
train_pd = crm.predict_pd(X_train_df)
test_pd = crm.predict_pd(X_test_df)

# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------
auc = roc_auc_score(y_test, test_pd)
ks = ks_statistic(y_test, test_pd)
gini = gini_coefficient(y_test, test_pd)

st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{auc:.3f}")
c2.metric("KS Statistic", f"{ks:.3f}")
c3.metric("Gini Coefficient", f"{gini:.3f}")

# --------------------------------------------------
# MODEL STABILITY (PSI — CORRECT)
# --------------------------------------------------
psi = population_stability_index(
    expected=pd.Series(train_pd),
    actual=pd.Series(test_pd)
)

st.metric(
    "Population Stability Index (PSI)",
    f"{psi:.3f}",
    help="PSI < 0.10: Stable | 0.10–0.25: Moderate Shift | > 0.25: Unstable"
)

st.divider()

# --------------------------------------------------
# PORTFOLIO RISK OVERVIEW
# --------------------------------------------------
st.subheader("📈 Portfolio Risk Overview")

portfolio_pd = crm.predict_pd(df[feature_cols])
portfolio_df, summary, bucket_dist = portfolio_metrics(
    df, portfolio_pd, lgd
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Exposure (EAD)", f"₹ {summary['Total_Exposure']:,.0f}")
c2.metric("Average Portfolio PD", f"{summary['Average_PD']:.2%}")
c3.metric("Total Expected Loss", f"₹ {summary['Total_Expected_Loss']:,.0f}")
c4.metric("High Risk Accounts", summary["High_Risk_Count"])

# Risk bucket distribution
fig, ax = plt.subplots()
bucket_dist.plot(kind="bar", ax=ax)
ax.set_ylabel("Number of Borrowers")
ax.set_title("Portfolio Risk Distribution")
st.pyplot(fig)

# Portfolio table
st.subheader("📋 Portfolio Details")
st.dataframe(
    portfolio_df[
        ["income", "loan_amount", "credit_score", "PD", "Risk_Bucket", "Expected_Loss"]
    ].style.format({
        "PD": "{:.2%}",
        "Expected_Loss": "₹ {:,.0f}"
    }),
    height=320
)

st.divider()

# --------------------------------------------------
# INDIVIDUAL BORROWER ASSESSMENT
# --------------------------------------------------
st.subheader("🔍 Individual Credit Assessment")

with st.form("borrower_form"):
    income = st.number_input("Annual Income", 20000, 500000, 60000)
    loan_amt = st.number_input("Loan Amount", 50000, 500000, 200000)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    age = st.slider("Age", 18, 75, 35)
    existing_loans = st.slider("Existing Loans", 0, 5, 1)

    submitted = st.form_submit_button("Evaluate Credit Risk")

if submitted:
    borrower_df = pd.DataFrame(
        [[income, loan_amt, credit_score, age, existing_loans]],
        columns=feature_cols
    )

    pd_value = crm.predict_pd(borrower_df)[0]
    bucket = risk_bucket(pd_value)
    decision = credit_decision(pd_value)

    expected_loss = pd_value * lgd * loan_amt
    rwa = approximate_rwa(pd_value, loan_amt)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Default (PD)", f"{pd_value:.2%}")
    c2.metric("Risk Bucket", bucket)
    c3.metric("Credit Decision", decision)
    c4.metric("Expected Loss", f"₹ {expected_loss:,.0f}")

    st.metric("Approximate RWA (Capital Intuition)", f"₹ {rwa:,.0f}")

    # Explainability (Logistic only)
    if model_type == "logistic":
        st.subheader("📉 Key Risk Drivers (Explainability)")
        explanation = logistic_explain(crm.model, feature_cols)
        st.dataframe(explanation)

# --------------------------------------------------
# FOOTNOTE
# --------------------------------------------------
st.markdown("""
### 📘 Risk Framework Notes
- **PD** estimated using statistical / ML credit risk models  
- **Expected Loss = PD × LGD × EAD**  
- **PSI** compares PD distributions (Train vs Test) to detect population shift  
- **Credit Decisions** follow simplified policy rules  
- **RWA** shown for capital impact intuition  

This structure mirrors real-world wholesale credit risk governance.
""")
