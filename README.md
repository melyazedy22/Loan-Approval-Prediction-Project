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
I implemented a systematic training pipeline where **each algorithm was evaluated in two stages**:
1.  **Baseline Model:** Trained using default parameters (without `GridSearchCV`) to establish a benchmark.
2.  **Tuned Model:** Optimized using `GridSearchCV` to find the best hyperparameters via Cross-Validation.

The following models were implemented and tracked using **MLflow**:

* **Logistic Regression**
    * *Baseline:* Default settings.
    * *GridSearch:* Tuned L1 (Lasso) and L2 (Ridge) regularization strength (`C`).
* **Decision Tree Classifier**
    * *Baseline:* Default growth.
    * *GridSearch:* Tuned `max_depth` and split criteria (Gini vs. Entropy).
* **Random Forest Classifier**
    * *Baseline:* Default ensemble settings.
    * *GridSearch:* Tuned number of trees (`n_estimators`) and tree depth (`max_depth`).
* **AdaBoost Classifier**
    * *Baseline:* Default boosting.
    * *GridSearch:* Tuned learning rate and number of estimators.

# 📊 Model Performance Comparison 

| Rank | Run Name                          | Run Type    | Accuracy | Precision | Recall | F1-Score | Duration (s) |
| :--- | :-------------------------------- | :---------- | -------: | --------: | -----: | -------: | -----------: |
| 1    | Logistic_Regression_L1_GridSearch | GridSearch  | **84.9%** |   82.8%   | **98.4%** | **89.9%** |      4.7     |
| 2    | SVM                               | Base        | **84.9%** |   82.8%   | **98.4%** | **89.9%** |      4.7     |
| 3    | AdaBoost                          | Base        | **84.9%** |   82.8%   | **98.4%** | **89.9%** |      4.6     |
| 4    | Decision_Tree                     | Base        | **84.9%** |   82.8%   | **98.4%** | **89.9%** |      4.8     |
| 5    | Random_Forest                     | Base        |   82.7%   |   83.2%   |  93.7% |   88.1%   |      5.0     |
| 6    | Logistic_Regression_L2_GridSearch | GridSearch  |   81.6%   |   86.0%   |  87.4% |   86.7%   |      4.6     |
| 7    | Logistic_Regression_L1_Tuned      | Tuned       |   80.5%   |   85.8%   |  85.8% |   85.8%   |      4.6     |
| 8    | Logistic_Regression_L2_Tuned      | Tuned       |   80.5%   |   85.8%   |  85.8% |   85.8%   |      4.7     |
| 9    | Logistic_Regression               | Base        |   80.5%   |   85.8%   |  85.8% |   85.8%   |      7.8     |
| 10   | Random_Forest_GridSearch          | GridSearch  |   76.2%   |   84.3%   |  80.3% |   82.3%   |      5.2     |
| 11   | Decision_Tree_GridSearch          | GridSearch  |   74.1%   |   83.2%   |  78.0% |   80.5%   |     11.4     |
| 12   | SVM_GridSearch                    | GridSearch  |   73.5%   |   79.1%   |  83.5% |   81.2%   |      4.4     |
| 13   | AdaBoost_GridSearch               | GridSearch  |   68.1%   |   82.1%   |  68.5% |   74.7%   |      5.2     |

**Notes:**
- Ranked by **Accuracy** descending (ties broken by F1-Score, then Recall).
- Top 4 models achieved identical best performance: **84.9% Accuracy** and **89.9% F1-Score**.
- Duration values taken directly from the latest run logs.


> **Key Insight:** While **Random Forest** provided strong results (~76%), the simpler **Logistic Regression (L1 Lasso)** achieved the highest accuracy (~82%) and generalization, likely due to the linear separability of the processed features.
## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/melyazedy22/loan-approval-prediction.git]
    cd Loan-Approval-Prediction-Project
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
