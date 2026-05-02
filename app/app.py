import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Approval System", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/model.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("ML-based decision support system")

st.divider()

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

income = st.sidebar.number_input("Applicant Income", min_value=0.0)
co_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0)
loan = st.sidebar.number_input("Loan Amount", min_value=0.0)
term = st.sidebar.number_input("Loan Term", min_value=0.0)
credit = st.sidebar.selectbox("Credit History", [1, 0])

predict_btn = st.sidebar.button("🚀 Predict")

# -----------------------------
# PREDICTION
# -----------------------------
if predict_btn:

    if income == 0 or loan == 0:
        st.warning("Enter valid values")
    else:
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
            "Property_Area": area
        }])

        result = model.predict(input_df)[0]

        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = None

        col1, col2 = st.columns(2)

        with col1:
            if result == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

        with col2:
            if prob:
                st.metric("Approval Probability", f"{prob:.2%}")
                st.progress(float(prob))

        # -----------------------------
        # EXPLANATION
        # -----------------------------
        st.subheader("🧠 Insights")

        if credit == 1:
            st.write("✔ Good credit history improves approval")
        else:
            st.write("❌ Poor credit history reduces approval")

        if income > loan * 10:
            st.write("✔ Strong income-to-loan ratio")
        else:
            st.write("⚠️ Weak income-to-loan ratio")

        # -----------------------------
        # DOWNLOAD
        # -----------------------------
        result_df = pd.DataFrame([{
            "Income": income,
            "Loan": loan,
            "Prediction": "Approved" if result == 1 else "Rejected"
        }])

        st.download_button(
            "📥 Download Result",
            result_df.to_csv(index=False),
            "prediction.csv"
        )

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.subheader("📊 Model Comparison")

try:
    results_df = pd.read_csv("outputs/results.csv")
    st.dataframe(results_df)

    best = results_df.sort_values("F1", ascending=False).iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Best Model", best["Model"])
    col2.metric("F1 Score", f"{best['F1']:.2f}")

except:
    st.info("Train model first")

# -----------------------------
# DATA VISUALIZATION
# -----------------------------
st.subheader("📈 Dataset Insights")

try:
    df = pd.read_csv("data/train.csv")

    fig, ax = plt.subplots()
    df.groupby("Credit_History")["Loan_Status"].value_counts().unstack().plot(kind="bar", ax=ax)
    st.pyplot(fig)

except:
    st.warning("Dataset not found")

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption("Built using ML Pipeline + Streamlit")