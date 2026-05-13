
import pandas as pd


def load_and_preprocess():

    # LOAD DATA
    df = pd.read_csv("data/train2.csv")

    # ==============================
    # HANDLE MISSING VALUES
    # ==============================
    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed"
    ]

    for col in categorical_cols:
        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    numerical_cols = [
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    for col in numerical_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # ==============================
    # REMOVE UNNECESSARY COLUMN
    # ==============================
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    # ==============================
    # DATA TRANSFORMATION
    # ==============================
    df["Dependents"] = df["Dependents"].replace(
        "3+",
        3
    )

    df["Dependents"] = df["Dependents"].astype(int)

    # ==============================
    # TARGET ENCODING
    # ==============================
    df["Loan_Status"] = df["Loan_Status"].map({
        "Y": 1,
        "N": 0
    })

    return df

