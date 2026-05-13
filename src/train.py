
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import load_and_preprocess

# =========================================
# LOAD DATA
# =========================================
df = load_and_preprocess()

print("Dataset Loaded Successfully")
print(df.head())

# =========================================
# REMOVE UNNECESSARY COLUMNS
# =========================================
X = df.drop(
    columns=[
        "Loan_Status",
        "Loan_Status_Label"
    ],
    errors="ignore"
)

y = df["Loan_Status"]

# =========================================
# COLUMN TYPES
# =========================================
categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# =========================================
# PREPROCESSOR
# =========================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# =========================================
# TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# MODELS
# =========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "KNN": KNeighborsClassifier(),

    "SVM": SVC(
        probability=True
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

# =========================================
# TRAINING LOOP
# =========================================
results = []

best_pipeline = None
best_model_name = ""
best_score = 0

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # TRAIN
    pipeline.fit(X_train, y_train)

    # PREDICT
    predictions = pipeline.predict(X_test)

    # METRICS
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    # SAVE RESULTS
    results.append([
        name,
        round(accuracy, 4),
        round(f1, 4)
    ])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # BEST MODEL
    if f1 > best_score:
        best_score = f1
        best_pipeline = pipeline
        best_model_name = name

# =========================================
# RESULTS DATAFRAME
# =========================================
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1"
    ]
)

# SAVE RESULTS
results_df.to_csv(
    "outputs/results.csv",
    index=False
)

# DISPLAY RESULTS
print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(results_df)

# =========================================
# BEST MODEL
# =========================================
print("\n===================================")
print(f"BEST MODEL: {best_model_name}")
print(f"BEST F1 SCORE: {best_score:.4f}")
print("===================================")

# =========================================
# SAVE FINAL MODEL
# =========================================
joblib.dump(
    best_pipeline,
    "models/model.pkl"
)

print("\n✅ Final pipeline model saved successfully!")

