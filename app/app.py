import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval System",
    layout="wide"
)

# -----------------------------
# LOAD MODEL (Pipeline)
# -----------------------------
model = joblib.load("models/model.pkl")

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
# PREDICTION LOGIC
# -----------------------------
if predict_btn:

    if income == 0 or loan == 0:
        st.warning("⚠️ Please enter valid income and loan amount")
        st.stop()

    # 🔥 Create dataframe (same as training)
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

    # 🔥 Predict (pipeline handles everything)
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
# FOOTER
# -----------------------------
st.divider()
st.caption("Built with Machine Learning (Scikit-learn Pipeline) and Streamlit")