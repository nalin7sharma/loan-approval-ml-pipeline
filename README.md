# 🏦 Loan Approval ML Pipeline

A complete end-to-end machine learning pipeline for predicting loan approval outcomes. This project covers data preprocessing, exploratory data analysis, multi-model training using scikit-learn Pipelines, model evaluation, and deployment via a Streamlit web application.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Models Trained](#models-trained)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Deployment](#deployment)
- [Future Scope](#future-scope)

---

## 📌 Project Overview

Financial institutions process thousands of loan applications every day. Manual review is time-consuming and inconsistent. This project builds a supervised binary classification pipeline that predicts whether a loan application should be **Approved** or **Rejected** based on applicant attributes.

The pipeline is built with clean, reproducible stages: raw data ingestion → preprocessing → EDA → feature engineering → model training & comparison → best-model selection → serialization → Streamlit deployment.

---

## 🗂 Project Structure

```
loan-approval-ml-pipeline/
│
├── data/
│   └── train2.csv               # Training dataset
│
├── models/
│   └── model.pkl                # Saved best model (Logistic Regression pipeline)
│
├── outputs/
│   └── results.csv              # Model comparison results
│
├── eda.ipynb                    # Main project notebook (EDA + training)
├── app.py                       # Streamlit web application (deployment)
└── README.md
```

---

## 📊 Dataset

The dataset (`train2.csv`) contains loan application records with the following features:

| Column | Type | Description |
|---|---|---|
| `Loan_ID` | Object | Unique identifier (dropped during preprocessing) |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Numerical | Number of dependents (0, 1, 2, 3+) |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Monthly income of the applicant |
| `CoapplicantIncome` | Numerical | Monthly income of the co-applicant |
| `LoanAmount` | Numerical | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Numerical | Term of the loan in months |
| `Credit_History` | Numerical | Credit history meets guidelines (1 = Yes, 0 = No) |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | Target | **Y (Approved) / N (Rejected)** |

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Framework | scikit-learn |
| Model Serialization | joblib |
| Deployment | Streamlit |

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/loan-approval-ml-pipeline.git
cd loan-approval-ml-pipeline
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

---

## 🚀 Usage

**Run the Jupyter Notebook**

Open `eda.ipynb` in Jupyter or VS Code and run all cells sequentially. The notebook will:

- Load and preprocess the dataset
- Generate EDA visualizations
- Train 5 classification models
- Display a model comparison table and charts
- Save the best model to `models/model.pkl`

**Run the Streamlit App**

```bash
streamlit run app.py
```

This launches an interactive web UI where you can enter applicant details and get an instant loan approval prediction.

---

## 🔧 Pipeline Architecture

Each model is wrapped inside a scikit-learn `Pipeline` with the same two-stage structure:

```
Raw Features
     │
     ▼
ColumnTransformer
 ├── StandardScaler        →  Numerical columns
 └── OneHotEncoder         →  Categorical columns
     │
     ▼
Classifier (LR / KNN / SVM / DT / RF)
     │
     ▼
Prediction (Approved / Rejected)
```

Using `Pipeline` ensures that:
- Preprocessing is fitted only on training data (no data leakage)
- The same transformations are automatically applied during inference
- The entire pipeline (preprocessor + model) is serialized as a single `.pkl` file

**Preprocessing steps applied:**

- `Loan_ID` column dropped (non-informative identifier)
- Missing categorical values filled with the **column mode**
- Missing numerical values (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) filled with the **column median**
- `Dependents` column: `'3+'` replaced with `3` and cast to integer
- `Loan_Status` target encoded as `1 (Approved)` / `0 (Rejected)`

---

## 🤖 Models Trained

Five classification algorithms were trained and evaluated using an 80/20 train-test split (`random_state=42`):

| # | Model | Key Hyperparameters |
|---|---|---|
| 1 | Logistic Regression | `max_iter=1000` |
| 2 | K-Nearest Neighbors (KNN) | Default (k=5) |
| 3 | Support Vector Machine (SVM) | `probability=True` |
| 4 | Decision Tree | Default |
| 5 | Random Forest | Default |

---

## 📈 Evaluation Metrics

Each model was evaluated on the held-out test set using:

**Accuracy** — overall percentage of correct predictions.

**F1 Score** — harmonic mean of Precision and Recall; useful for imbalanced class distributions.

**Confusion Matrix** — breakdown of True Positives, True Negatives, False Positives, and False Negatives (shown for the best model).

**Classification Report** — per-class Precision, Recall, and F1 Score.

---

## 🏆 Results

Model comparison across Accuracy and F1 Score:

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ✅ Best | ✅ Best |
| SVM | High | High |
| Random Forest | High | High |
| Decision Tree | Moderate | Moderate |
| KNN | Moderate | Moderate |

> **Logistic Regression** achieved the best overall performance and was selected as the final model. Its pipeline was saved to `models/model.pkl` using `joblib`.

EDA insights from the notebook:
- Applicants **with a positive Credit History** have significantly higher loan approval rates.
- The dataset shows a moderate class imbalance, with more approvals than rejections.
- `LoanAmount`, `ApplicantIncome`, and `Credit_History` are the most correlated features with the target.

---

## 🌐 Deployment

The final model is deployed as a **Streamlit web application**. The app:

- Accepts user input for all applicant features via an interactive form
- Loads the serialized `model.pkl` pipeline
- Runs the same preprocessing and prediction steps automatically
- Displays the predicted result — **Approved ✅** or **Rejected ❌**

To run locally:

```bash
streamlit run app.py
```

---

## 🔭 Future Scope

- **Hyperparameter Tuning** — Use `GridSearchCV` or `RandomizedSearchCV` to optimize model parameters
- **Advanced Models** — Experiment with XGBoost, LightGBM, or deep learning classifiers
- **Explainable AI** — Integrate SHAP or LIME to provide feature-level explanations per prediction
- **Cloud Deployment** — Deploy the Streamlit app on Heroku, AWS, or Streamlit Community Cloud
- **Real-time Database Integration** — Connect to a live database for real-time applicant data
- **CI/CD Pipeline** — Automate retraining and deployment with GitHub Actions

---

# Author {CSE-AI Student Chhatrapati Shahu Ji Maharaj University}
Rajyavardhan Radhey 
Nalin Sharma 
Amritansh Singh
Aman
