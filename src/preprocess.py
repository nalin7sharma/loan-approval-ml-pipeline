import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Approval System", layout="wide")

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "train.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "results.csv"

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("❌ Model not found. Please run training first.")
    st.stop()

# -----------------------------
# TITLE
# -----------------------------
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("Machine Learning powered decision support system")

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🧾 Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

income = st.sidebar.number_input("Applicant Income", min_value=0.0, step=100.0)
co_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0, step=100.0)
loan = st.sidebar.number_input("Loan Amount", min_value=0.0)
term = st.sidebar.number_input("Loan Term", min_value=0.0)
credit = st.sidebar.selectbox("Credit History", [1, 0])

predict_btn = st.sidebar.button("🚀 Predict")

# -----------------------------
# PREDICTION
# -----------------------------
if predict_btn:

    if income == 0 or loan == 0:
        st.warning("⚠️ Please enter valid income and loan values")
        st.stop()

    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_emp,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan,
        "Loan_Amount_Term": term,
        "Credit_History": credit,
        "Property_Area": property_area
    }])

    result = model.predict(input_df)[0]

    try:
        prob = model.predict_proba(input_df)[0][1]
    except:
        prob = None

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if result == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    with col2:
        if prob is not None:
            st.metric("Approval Probability", f"{prob:.2%}")
            st.progress(float(prob))

    # -----------------------------
    # INSIGHTS
    # -----------------------------
    st.subheader("🧠 Prediction Insights")

    if credit == 1:
        st.write("✔ Good credit history improves approval chances")
    else:
        st.write("❌ Poor credit history reduces approval chances")

    if income > loan * 10:
        st.write("✔ Strong income-to-loan ratio")
    else:
        st.write("⚠️ Weak income compared to loan amount")

    # -----------------------------
    # DOWNLOAD RESULT
    # -----------------------------
    result_df = pd.DataFrame([{
        "Gender": gender,
        "Income": income,
        "Loan": loan,
        "Prediction": "Approved" if result == 1 else "Rejected"
    }])

    st.download_button(
        label="📥 Download Result",
        data=result_df.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("📊 Model Comparison")

if RESULTS_PATH.exists():
    results_df = pd.read_csv(RESULTS_PATH)
    st.dataframe(results_df)

    best = results_df.sort_values("F1", ascending=False).iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Best Model", best["Model"])
    col2.metric("F1 Score", f"{best['F1']:.2f}")
else:
    st.info("Run training to see model comparison")

# -----------------------------
# DATASET INSIGHTS
# -----------------------------
st.subheader("📈 Dataset Insights")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    fig, ax = plt.subplots()
    df.groupby("Credit_History")["Loan_Status"].value_counts().unstack().plot(kind="bar", ax=ax)

    ax.set_title("Loan Approval vs Credit History")
    ax.set_xlabel("Credit History")
    ax.set_ylabel("Count")

    st.pyplot(fig)
else:
    st.warning("Dataset not found")
    st.write(f"Expected path: {DATA_PATH}")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Built using Machine Learning Pipeline + Streamlit")
