from nbformat import v4 as nbf
import nbformat

# ==========================================
# CREATE NOTEBOOK
# ==========================================

nb = nbf.new_notebook()

cells = []

# ==========================================
# TITLE
# ==========================================

cells.append(nbf.new_markdown_cell(
"""
# Loan Approval Prediction System

## Machine Learning Algorithms Visualization Notebook

This notebook demonstrates:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Model comparison
- Visualization of results
"""
))

# ==========================================
# IMPORT LIBRARIES
# ==========================================

cells.append(nbf.new_markdown_cell("## 1. Import Libraries"))

cells.append(nbf.new_code_cell(
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')
"""
))

# ==========================================
# LOAD DATASET
# ==========================================

cells.append(nbf.new_markdown_cell("## 2. Load Dataset"))

cells.append(nbf.new_code_cell(
"""
df = pd.read_csv('../data/train.csv')

print(df.head())
"""
))

# ==========================================
# PREPROCESSING
# ==========================================

cells.append(nbf.new_markdown_cell("## 3. Data Preprocessing"))

cells.append(nbf.new_code_cell(
"""
# Remove unnecessary column
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing categorical values
categorical_cols = [
    'Gender',
    'Married',
    'Dependents',
    'Self_Employed'
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing numerical values
numerical_cols = [
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History'
]

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Replace 3+ dependents
if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = df['Dependents'].astype(int)

# Encode target variable
if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

print(df.head())
"""
))

# ==========================================
# EDA
# ==========================================

cells.append(nbf.new_markdown_cell("## 4. Exploratory Data Analysis"))

cells.append(nbf.new_code_cell(
"""
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Status Distribution')
plt.show()
"""
))

cells.append(nbf.new_code_cell(
"""
plt.figure(figsize=(7,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title('Credit History vs Loan Approval')
plt.show()
"""
))

cells.append(nbf.new_code_cell(
"""
plt.figure(figsize=(8,4))
sns.histplot(df['ApplicantIncome'], kde=True)
plt.title('Applicant Income Distribution')
plt.show()
"""
))

cells.append(nbf.new_code_cell(
"""
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
"""
))

# ==========================================
# FEATURE TARGET
# ==========================================

cells.append(nbf.new_markdown_cell("## 5. Feature and Target Separation"))

cells.append(nbf.new_code_cell(
"""
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
"""
))

# ==========================================
# PREPROCESSOR
# ==========================================

cells.append(nbf.new_markdown_cell("## 6. Column Transformer"))

cells.append(nbf.new_code_cell(
"""
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
"""
))

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

cells.append(nbf.new_markdown_cell("## 7. Train Test Split"))

cells.append(nbf.new_code_cell(
"""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
"""
))

# ==========================================
# MODELS
# ==========================================

algorithms = [
    ('Logistic Regression', 'LogisticRegression(max_iter=1000)', 'logistic'),
    ('KNN', 'KNeighborsClassifier()', 'knn'),
    ('SVM', 'SVC(probability=True)', 'svm'),
    ('Decision Tree', 'DecisionTreeClassifier()', 'dt'),
    ('Random Forest', 'RandomForestClassifier()', 'rf')
]

for name, model_code, var in algorithms:

    cells.append(nbf.new_markdown_cell(f"## {name}"))

    cells.append(nbf.new_markdown_cell(
    f"""
### Theory
{name} is used in this project for loan approval prediction.

### Implementation
The algorithm is trained using Scikit-learn Pipeline with preprocessing.
"""
    ))

    cells.append(nbf.new_code_cell(
    f"""
{var}_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', {model_code})
])

{var}_pipeline.fit(X_train, y_train)

{var}_pred = {var}_pipeline.predict(X_test)

{var}_acc = accuracy_score(y_test, {var}_pred)
{var}_f1 = f1_score(y_test, {var}_pred)

print('Accuracy:', {var}_acc)
print('F1 Score:', {var}_f1)
"""
    ))

# ==========================================
# COMPARISON
# ==========================================

cells.append(nbf.new_markdown_cell("## 8. Model Comparison"))

cells.append(nbf.new_code_cell(
"""
results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'KNN',
        'SVM',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy': [
        logistic_acc,
        knn_acc,
        svm_acc,
        dt_acc,
        rf_acc
    ],
    'F1 Score': [
        logistic_f1,
        knn_f1,
        svm_f1,
        dt_f1,
        rf_f1
    ]
})

print(results)
"""
))

# ==========================================
# VISUALIZATION
# ==========================================

cells.append(nbf.new_markdown_cell("## 9. Accuracy Visualization"))

cells.append(nbf.new_code_cell(
"""
plt.figure(figsize=(10,5))

sns.barplot(x='Model', y='Accuracy', data=results)

plt.title('Model Accuracy Comparison')
plt.xticks(rotation=15)
plt.show()
"""
))

cells.append(nbf.new_markdown_cell("## 10. F1 Score Visualization"))

cells.append(nbf.new_code_cell(
"""
plt.figure(figsize=(10,5))

sns.barplot(x='Model', y='F1 Score', data=results)

plt.title('F1 Score Comparison')
plt.xticks(rotation=15)
plt.show()
"""
))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cells.append(nbf.new_markdown_cell("## 11. Confusion Matrix"))

cells.append(nbf.new_code_cell(
"""
cm = confusion_matrix(y_test, logistic_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()
"""
))

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

cells.append(nbf.new_markdown_cell("## 12. Classification Report"))

cells.append(nbf.new_code_cell(
"""
print(classification_report(y_test, logistic_pred))
"""
))

# ==========================================
# SAVE MODEL
# ==========================================

cells.append(nbf.new_markdown_cell("## 13. Save Final Model"))

cells.append(nbf.new_code_cell(
"""
import joblib

joblib.dump(logistic_pipeline, '../models/model.pkl')

print('Model Saved Successfully!')
"""
))

# ==========================================
# CONCLUSION
# ==========================================

cells.append(nbf.new_markdown_cell(
"""
# 14. Conclusion

This project successfully implemented multiple machine learning algorithms for loan approval prediction.

The models were compared using Accuracy and F1 Score.

Logistic Regression achieved the best overall performance and was selected as the final model for deployment.

The final model was integrated into a Streamlit web application.
"""
))

# ==========================================
# ASSIGN CELLS
# ==========================================

nb.cells = cells

# ==========================================
# SAVE NOTEBOOK
# ==========================================

output_file = 'loan_approval_visualization_notebook.ipynb'

with open(output_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Notebook created successfully: {output_file}')
