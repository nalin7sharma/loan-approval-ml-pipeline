import pandas as pd

def load_and_preprocess():
    # load dataset
    df = pd.read_csv(r"C:\Users\LENOVO\.dev\loan-approval-ml-pipeline\data\train2.csv")

    # clean column names (very important)
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns)

    # check if target exists
    if "Loan_Status" not in df.columns:
        raise ValueError("❌ Loan_Status column not found. Please use correct dataset.")

    # drop ID
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    # ----------------------------
    # HANDLE MISSING VALUES
    # ----------------------------

    categorical_cols = [
        "Gender", "Married", "Dependents",
        "Education", "Self_Employed", "Property_Area"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    numerical_cols = [
        "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History"
    ]

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ----------------------------
    # CLEAN SPECIAL CASES
    # ----------------------------

    # Dependents column (3+ → 3)
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")
        df["Dependents"] = df["Dependents"].astype(int)

    # ----------------------------
    # ENCODE TARGET VARIABLE
    # ----------------------------

    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    print("✅ Preprocessing completed")
    print(df.head())

    return df