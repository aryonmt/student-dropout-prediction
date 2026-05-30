# 🎓 Student Dropout Prediction: An End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-blue.svg)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)

## 📌 Project Overview

Student attrition (dropout) is a critical challenge for higher education institutions, impacting university rankings, financial stability, and most importantly, students' career trajectories.

The goal of this project is to build a highly interpretable, production-ready Machine Learning pipeline to identify students at risk of dropping out before it happens. By shifting from experimental notebook scripts to a **modular, scalable software architecture**, this project demonstrates applied software engineering principles within a Data Science workflow.

## 📊 The Dataset

The dataset contains records of undergraduate students, encompassing a comprehensive set of variables collected at the time of enrollment and at the end of their first and second semesters.

**Key Feature Categories:**

- **Demographics & Socioeconomics:** Age, marital status, nationality, parents' qualifications, and occupation.
- **Macroeconomics:** Unemployment rate, inflation rate, and GDP at the time of enrollment.
- **Academic Performance:** Curricular units enrolled, evaluated, approved, and grades for the 1st and 2nd semesters.
- **Target Variable:** A binary classification indicating the student's final status (`Graduate` vs. `Dropout`).

_Note: The dataset exhibits a moderate class imbalance (approx. 60.9% Graduates vs. 39.1% Dropouts)._

## 🏗️ Architecture & Methodology

To ensure high maintainability and prevent common ML pitfalls (e.g., Data Leakage, Overfitting), the project is strictly decoupled into functional modules:

1. **`data_loader.py`**: Handles data ingestion, cleaning, and Stratified splitting to maintain class proportions.
2. **`features.py`**: A rigorous, data-leakage-free Feature Selection pipeline. To combat severe multicollinearity (e.g., correlation between 1st and 2nd-semester grades), it combines **Mutual Information (MI)** and **L1 Regularization (Lasso)** exclusively on the training set to extract the most robust predictors.
3. **`train.py`**: Utilizes **Optuna** for automated, objective hyperparameter tuning. The objective function maximizes the `ROC-AUC` score across Stratified 5-Fold Cross-Validation, ensuring generalized performance.
4. **`evaluate.py`**: Calculates final metrics, plots the ROC Curve, and generates Explainable AI (XAI) insights using **SHAP**.

## 🚀 Key Insights & Explainability (SHAP)

Instead of treating the XGBoost model as a black box, we used SHAP values to decode its decision-making logic. The model successfully learned causal, real-world patterns rather than memorizing noise:

- **Academic Success:** High approval rates in 1st and 2nd-semester curricular units heavily push the prediction toward `Graduate`.
- **Financial Stability:** The `Tuition fees up to date` feature is a massive indicator. Unpaid fees strongly correlate with an impending `Dropout`.
- **Age Factor:** Older students at the time of enrollment show a slightly higher risk of dropping out compared to standard-age entrants.

## 📈 Model Performance

The optimized XGBoost model achieved outstanding generalization on the unseen test set, proving highly effective at balancing precision and recall for the minority class (Dropouts):

- **ROC-AUC Score:** `0.9671`
- **F1-Score (Macro):** `0.92`
- **Dropout Recall:** `0.88` (Successfully identifying 88% of actual dropouts)
- **Dropout Precision:** `0.91` (When predicting a dropout, it is correct 91% of the time)
- **Overall Accuracy:** `0.9219`

## ⚙️ Installation & Usage

**1. Clone the repository:**

```bash
git clone [https://github.com/aryonmt/dropout_prediction.git](https://github.com/aryonmt/dropout_prediction.git)
cd dropout_prediction
```

**2. Set up the virtual environment:**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Install dependencies:**

Ensure the raw dataset is located at data/data.csv, then run:

```bash
python main.py
```

Outputs, including the ROC curve and SHAP summary plots, will be automatically saved in the reports/ directory.

## 📁 Repository Structure

dropout_prediction/
├── data/
│ └── data.csv
├── notebooks/
│ └── 01_EDA.ipynb # Deep dive into Data Quality & Multicollinearity
├── reports/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ └── shap_summary.png
├── src/
│ ├── **init**.py
│ ├── config.py # Centralized constants & paths
│ ├── data_loader.py
│ ├── features.py
│ ├── train.py
│ └── evaluate.py
├── main.py # Pipeline execution entry point
├── requirements.txt
└── README.md
