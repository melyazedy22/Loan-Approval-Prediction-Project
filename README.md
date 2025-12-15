# 📊 Loan Approval Prediction – Machine Learning Project

This project demonstrates a complete **machine learning pipeline** to predict whether a loan application will be **approved or rejected** based on applicant information.

The notebook covers data understanding, preprocessing, model training, evaluation, and performance analysis using real-world inspired loan data.

---

## 🧠 Project Objective

Build and evaluate machine learning models that can accurately predict **Loan Status** (`Approved / Not Approved`) based on customer and loan-related features.

This type of problem is a **binary classification task**, commonly used in banking and financial decision systems.

---

## 📁 Dataset Overview

The dataset includes information such as:

* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Gender
* Education
* Self Employment
* Property Area
* Marital Status

🎯 **Target Variable**:

* `Loan_Status` (0 = Not Approved, 1 = Approved)

---

## 🔄 Workflow & Pipeline

The notebook follows these key steps:

### 1️⃣ Data Exploration (EDA)

* Inspect dataset structure
* Check missing values
* Analyze target class distribution
* Understand feature types (numerical & categorical)

### 2️⃣ Data Preprocessing

* Handling missing values
* Encoding categorical features

  * Label Encoding
  * One-Hot Encoding
* Feature scaling using **StandardScaler**
* Splitting data into training and testing sets

### 3️⃣ Model Building

Several classification models are trained and evaluated, such as:

* Logistic Regression
* Decision Tree Classifier
* (Optional) Other ML models for comparison

### 4️⃣ Model Evaluation

Models are evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report
* Cross-Validation Scores

Threshold tuning is also applied to improve model decision quality.

---

## 📈 Results & Insights

* Model performance is compared using validation metrics
* Overfitting and underfitting are analyzed
* Best-performing model is identified based on evaluation scores

---

## 🛠️ Technologies & Libraries

The project is implemented using **Python** and the following libraries:

* `NumPy`
* `Pandas`
* `Matplotlib`
* `Seaborn`
* `Scikit-learn`

---

## ▶️ How to Run the Notebook

1. Clone or download the repository
2. Make sure Python (>=3.8) is installed
3. Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

4. Open the notebook:

```bash
jupyter notebook ML_Project_Predicting-Loan-Approval.ipynb
```

5. Run cells sequentially from top to bottom

---

## 📌 Notes

* This project is intended for **educational purposes**
* Dataset structure is similar to real-world loan approval systems
* Feature engineering and threshold tuning can further improve results

---

## 👤 Author

**Mahmoud Elyazedy**
Faculty of Engineering – Computer & Automatic Control
Interested in Data Science & Machine Learning

---

✅ *Feel free to improve the model, try new algorithms, or enhance feature engineering!*
