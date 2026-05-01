import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_and_preprocess

# load data
df = load_and_preprocess()

# encode categorical
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = []

best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, acc, f1])

    if f1 > best_score:
        best_score = f1
        best_model = model

# save results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
results_df.to_csv("outputs/results.csv", index=False)

print(results_df)

# save best model
pickle.dump(best_model, open("models/model.pkl", "wb"))

print("Best model saved!")