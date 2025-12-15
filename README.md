# 🏠 Loan Approval Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This machine learning project aims to automate the loan eligibility process for financial institutions. By analyzing customer details provided in online application forms, the model predicts whether a loan should be **Approved (`Y`)** or **Rejected (`N`)**. 

The solution helps streamline decision-making, reduce manual effort, and manage risk effectively by identifying eligible applicants based on their financial and demographic profiles.

## 📂 Dataset Description
The dataset consists of **614 records** and **13 features**, covering demographic and financial information.

| Feature | Description |
| :--- | :--- |
| **Loan_Status** | Target Variable (Y: Approved, N: Rejected) |
| **Loan_ID** | Unique Loan ID |
| **Gender** | Male / Female |
| **Married** | Applicant married (Y/N) |
| **Dependents** | Number of dependents |
| **Education** | Graduate / Under Graduate |
| **Self_Employed** | Self-employed (Y/N) |
| **ApplicantIncome** | Income of the applicant |
| **CoapplicantIncome** | Income of the co-applicant |
| **LoanAmount** | Loan amount in thousands |
| **Loan_Amount_Term** | Term of loan in months |
| **Credit_History** | Credit history meets guidelines (1: Yes, 0: No) |
| **Property_Area** | Urban / Semi Urban / Rural |

## 🛠️ Workflow & Methodology

### 1. Data Preprocessing
* **Missing Values:** Imputed categorical variables with **Mode** and numerical variables with **Median** to ensure data completeness.
* **Encoding:** Applied **One-Hot Encoding** for categorical features and **Label Encoding** for the target variable to make them machine-readable.
* **Balancing:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in the training data, ensuring the model doesn't become biased toward the majority class.
* **Scaling:** Normalized features using `StandardScaler` to bring all variables to a similar scale.

### 2. Feature Engineering
* **Total Income:** Created a new feature combining `ApplicantIncome` and `CoapplicantIncome` to capture the full financial picture.
* **Log Transformation:** Applied to `LoanAmount` and `TotalIncome` to handle skewness and normalize distributions, improving model performance.
* **Feature Selection:** Dropped redundant columns (e.g., original Income columns, Loan ID) after engineering to reduce noise.

### 3. Model Training & Tracking
Several machine learning models were implemented and tracked using **MLflow** to monitor experiments and configuration:
* **Logistic Regression** (Baseline, L1 Lasso, and L2 Ridge Regularization)
* **Decision Tree Classifier** (Tested with Gini and Entropy criteria, and tuned Max Depth)
* **AdaBoost Classifier** (Optimized via GridSearch for ensemble learning)

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/loan-approval-prediction.git](https://github.com/yourusername/loan-approval-prediction.git)
    cd loan-approval-prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn mlflow imbalanced-learn
    ```

3.  **Run the Notebook**
    ```bash
    jupyter notebook ML_Project_Predicting-Loan-Approval.ipynb
    ```

4.  **View MLflow Experiments** (Optional)
    To view the experiment logs and artifacts:
    ```bash
    mlflow ui
    ```

## ⚙️ Technologies Used
* **Python 3.x**
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-Learn** (Model Training & Evaluation)
* **Imbalanced-Learn** (SMOTE for balancing data)
* **MLflow** (Experiment Tracking)

## 📜 Conclusion
This project demonstrates a complete machine learning pipeline, from data cleaning and feature engineering to model training. The use of advanced techniques like SMOTE and hyperparameter tuning ensures a robust solution for the loan approval prediction problem.
