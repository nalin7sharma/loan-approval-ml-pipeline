import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_and_preprocess()

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -----------------------------
# COLUMN TYPES
# -----------------------------
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# -----------------------------
# PREPROCESSOR
# -----------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = []
best_pipeline = None
best_score = 0

# -----------------------------
# TRAIN LOOP
# -----------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, acc, f1])

    if f1 > best_score:
        best_score = f1
        best_pipeline = pipeline
        best_model_name = name

# -----------------------------
# SAVE RESULTS
# -----------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
results_df.to_csv("outputs/results.csv", index=False)

print(results_df)
print(f"\nBest Model: {best_model_name}")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(best_pipeline, "models/model.pkl")

print("✅ Final pipeline model saved!")