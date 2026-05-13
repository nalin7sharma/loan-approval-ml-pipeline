
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================
try:
    model = joblib.load("models/model.pkl")
except:
    st.error("❌ Model not found. Train the model first.")
    st.stop()

# =========================================
# TITLE
# =========================================
st.title("🏦 Loan Approval Prediction Dashboard")
st.markdown("### Machine Learning Based Loan Approval System")

st.divider()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.header("🧾 Applicant Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_emp = st.sidebar.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

income = st.sidebar.number_input(
    "Applicant Income",
    min_value=0.0,
    step=100.0
)

co_income = st.sidebar.number_input(
    "Coapplicant Income",
    min_value=0.0,
    step=100.0
)

loan = st.sidebar.number_input(
    "Loan Amount",
    min_value=0.0,
    step=1.0
)

term = st.sidebar.number_input(
    "Loan Amount Term",
    min_value=0.0,
    step=12.0,
    value=360.0
)

credit = st.sidebar.selectbox(
    "Credit History",
    [1, 0]
)

predict_btn = st.sidebar.button("🚀 Predict Loan Status")

# =========================================
# PREDICTION
# =========================================
if predict_btn:

    if income <= 0 or loan <= 0:
        st.warning("⚠️ Please enter valid Applicant Income and Loan Amount")

    else:

        # INPUT DATAFRAME
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": str(dependents),
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": income,
            "CoapplicantIncome": co_income,
            "LoanAmount": loan,
            "Loan_Amount_Term": term,
            "Credit_History": credit,
            "Property_Area": property_area
        }])

        try:

            # PREDICT
            result = model.predict(input_df)[0]

            # PROBABILITY
            try:
                probability = model.predict_proba(input_df)[0][1]
            except:
                probability = 0

            st.divider()

            col1, col2 = st.columns(2)

            # =========================================
            # RESULT
            # =========================================
            with col1:

                if result == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")

            # =========================================
            # PROBABILITY
            # =========================================
            with col2:

                st.metric(
                    "Approval Probability",
                    f"{probability:.2%}"
                )

                st.progress(float(probability))

            # =========================================
            # INSIGHTS
            # =========================================
            st.subheader("🧠 Prediction Insights")

            if credit == 1:
                st.write("✔ Strong credit history positively influenced prediction.")
            else:
                st.write("❌ Poor credit history negatively affected prediction.")

            if income > loan * 10:
                st.write("✔ Applicant has a strong income-to-loan ratio.")
            else:
                st.write("⚠️ Income-to-loan ratio is relatively low.")

            if self_emp == "Yes":
                st.write("ℹ Applicant is self-employed.")

            if property_area == "Urban":
                st.write("ℹ Urban property area may improve approval chances.")

            # =========================================
            # DOWNLOAD RESULT
            # =========================================
            st.subheader("📥 Download Prediction")

            result_df = pd.DataFrame([{
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
                "Property_Area": property_area,
                "Prediction": "Approved" if result == 1 else "Rejected",
                "Probability": probability
            }])

            st.download_button(
                label="📄 Download Result CSV",
                data=result_df.to_csv(index=False),
                file_name="loan_prediction_result.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =========================================
# MODEL COMPARISON
# =========================================
st.divider()

st.subheader("📊 Model Performance Comparison")

try:

    results_df = pd.read_csv("outputs/results.csv")

    st.dataframe(results_df)

    best_model = results_df.sort_values(
        by="F1",
        ascending=False
    ).iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Best Model",
            best_model["Model"]
        )

    with col2:
        st.metric(
            "Best F1 Score",
            f"{best_model['F1']:.2f}"
        )

    # BAR CHART
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

except:
    st.warning("⚠️ results.csv not found")

# =========================================
# DATASET VISUALIZATION
# =========================================
st.divider()

st.subheader("📈 Dataset Insights")

try:

    df = pd.read_csv("data/train2.csv")

    # FIX TARGET LABEL
    df["Loan_Status_Label"] = df["Loan_Status"].map({
        "Y": "Approved",
        "N": "Rejected"
    })

    col1, col2 = st.columns(2)

    # =========================================
    # LOAN STATUS DISTRIBUTION
    # =========================================
    with col1:

        fig1, ax1 = plt.subplots(figsize=(5, 4))

        sns.countplot(
            data=df,
            x="Loan_Status_Label",
            ax=ax1
        )

        plt.title("Loan Status Distribution")

        st.pyplot(fig1)

    # =========================================
    # CREDIT HISTORY ANALYSIS
    # =========================================
    with col2:

        fig2, ax2 = plt.subplots(figsize=(5, 4))

        sns.countplot(
            data=df,
            x="Credit_History",
            hue="Loan_Status_Label",
            ax=ax2
        )

        plt.title("Credit History vs Loan Approval")

        st.pyplot(fig2)

except Exception as e:
    st.warning(f"Dataset Error: {e}")

# =========================================
# FOOTER
# =========================================
st.divider()

st.caption(
    "Built using Python, Scikit-learn, Streamlit, Pandas, and Machine Learning Pipeline"
)

