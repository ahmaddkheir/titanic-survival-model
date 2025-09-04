

# Titanic Survival Model

Predict whether a passenger survived the Titanic disaster using **scikit-learn**.
This project demonstrates a complete ML workflow: data preprocessing, feature engineering, model training, evaluation, and exporting a trained pipeline for inference.

---

## Overview

* **Goal:** Predict survival of Titanic passengers (`Survived`: 0 = did not survive, 1 = survived).
* **Dataset:** Titanic dataset (train/test CSVs).
* **Approach:**

  * Exploratory Data Analysis (EDA).
  * Feature engineering (e.g., extract `Title` from `Name`, impute missing `Age`, encode categoricals).
  * Preprocessing with `ColumnTransformer` and `Pipeline`.
  * Model comparison: Logistic Regression, Random Forest, XGBoost.
  * Hyperparameter tuning with `GridSearchCV`.
* **Final model:** Logistic Regression (`C=10`, `penalty=l2`, `solver=lbfgs`).

---

## 📂 Project Structure

```plaintext
titanic-survival-model/
│── data/
│   ├── Titanic-Dataset.csv     # training data
│   └── titanic_test.csv        # test data
│
│── notebooks/
│   └── main.ipynb              # Jupyter notebook (EDA + training)
│
│── models/
│   └── save_model.py           # helper functions to save/load pipeline
│
│── outputs/
│   └── submission.csv          # predictions for test set
│
│── .gitignore
│── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1) Clone the repository

```bash
git clone https://github.com/ahmaddkheir/titanic-survival-model.git
cd titanic-survival-model
```

### 2) Create and activate a virtual environment

**Mac/Linux**

```bash
python -m venv virtual
source virtual/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv virtual
virtual\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Launch the notebook and run end-to-end

```bash
jupyter notebook notebooks/main.ipynb
```

Run all cells to:

* explore and clean the data,
* engineer features,
* train and tune models,
* export the best pipeline,
* generate `outputs/submission.csv`.

---

## Tools and Libraries

* **Python:** 3.10
* **Core:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **ML:** `scikit-learn`
* **Boosting (optional):** `xgboost`

---


