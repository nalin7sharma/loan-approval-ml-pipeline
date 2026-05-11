# Loan Approval ML Pipeline

A complete end-to-end Machine Learning project for predicting loan approval status using applicant information. This project demonstrates the implementation of a full ML pipeline including data preprocessing, model training, evaluation, prediction, and deployment using Streamlit.

---

# Project Overview

The Loan Approval ML Pipeline project is designed to automate the process of loan approval prediction using machine learning algorithms. The system analyzes applicant details such as income, education, credit history, and loan amount to predict whether the loan application should be approved or rejected.

The project follows a complete machine learning workflow:

- Data Collection
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Deployment

The frontend of the project is built using Streamlit, allowing users to interact with the model in real time.

---

# Features

- End-to-End Machine Learning Pipeline
- Data Cleaning and Preprocessing
- Feature Encoding
- Model Training and Evaluation
- Real-Time Loan Prediction
- Streamlit Web Application
- Saved Trained Model using Pickle
- Easy Deployment and Scalability

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Streamlit | Frontend Web Application |
| Pickle | Model Serialization |
| Jupyter Notebook | Exploratory Data Analysis |

---

# Project Structure

```bash
loan-approval-ml-pipeline-main/
│
├── app/
│   └── app.py
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── train2.csv
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   └── results.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# Working of the Project

## Step 1: Dataset Collection

The dataset containing applicant details is loaded from CSV files.

## Step 2: Data Preprocessing

The preprocessing stage handles:

- Missing values
- Categorical encoding
- Data cleaning
- Feature transformation

## Step 3: Model Training

Machine learning algorithms are trained using historical loan approval data.

## Step 4: Model Evaluation

The trained model is evaluated using performance metrics such as:

- Accuracy
- Precision
- Recall
- Confusion Matrix

## Step 5: Model Saving

The best trained model is saved using Pickle (`model.pkl`).

## Step 6: Streamlit Deployment

A Streamlit-based frontend is used to provide real-time predictions.

---

# Dataset Features

| Feature | Description |
|----------|-------------|
| Gender | Applicant Gender |
| Married | Marital Status |
| Dependents | Number of Dependents |
| Education | Education Qualification |
| Self_Employed | Employment Status |
| ApplicantIncome | Applicant Income |
| CoapplicantIncome | Co-applicant Income |
| LoanAmount | Loan Amount |
| Loan_Amount_Term | Loan Duration |
| Credit_History | Credit History |
| Property_Area | Property Location |
| Loan_Status | Approval Status |

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/loan-approval-ml-pipeline.git
```

## Navigate to Project Directory

```bash
cd loan-approval-ml-pipeline
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Training Script

```bash
python src/train.py
```

## Run Prediction Script

```bash
python src/predict.py
```

## Launch Streamlit Application

```bash
streamlit run app/app.py
```

---

# Example Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Encoding
    ↓
Trained ML Model
    ↓
Prediction Output
```

---

# Output

The system predicts:

- Loan Approved
- Loan Rejected

based on user input parameters.

---

# Advantages

- Faster Loan Processing
- Reduced Manual Effort
- Automated Predictions
- User-Friendly Interface
- Scalable Architecture

---

# Limitations

- Dependent on Dataset Quality
- Requires Model Retraining
- Prediction Accuracy Depends on Training Data

---

# Future Improvements

- Cloud Deployment
- Deep Learning Integration
- Database Connectivity
- Mobile Application Support
- Real-Time Banking Integration

---

# Author

Rajyavardhan Radhey

Nalin Sharma

Amritansh

Aman

CSE-AI Student  
Chhatrapati Shahu Ji Maharaj University

---

# License

This project is created for educational and learning purposes.

---

# References

- Scikit-learn Documentation
- Streamlit Documentation
- Pandas Documentation
- Python Official Website
