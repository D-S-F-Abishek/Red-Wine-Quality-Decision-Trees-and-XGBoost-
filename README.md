# 🍷 Red Wine Quality Prediction using Decision Trees & XGBoost

A **hands-on Machine Learning project** built after completing **Decision Trees and Tree Ensembles (XGBoost)** . This repository demonstrates how to move from ML theory to a complete, working project.

---

## 📌 Project Description

The objective of this project is to **predict the quality of red wine** based on physicochemical features using **tree-based models**. The project follows a clean, end-to-end ML workflow suitable for **beginners transitioning into practical ML projects**.

---

## 🎯 Goals

* Apply Decision Trees on a real-world dataset
* Understand ensemble learning with XGBoost
* Perform EDA and feature analysis
* Evaluate and compare ML models
* Build confidence in ML project structuring

---

## 📊 Dataset

* **Name:** Red Wine Quality Dataset
* **Source:** UCI Machine Learning Repository
* **Problem Type:** Supervised Learning
* **Target Variable:** `quality`

### Features

```
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
```

---

## 🏗️ Repository Structure

```
RedWineQualityPrediction/
│
├── RedWineDecisionTrees.ipynb   # Main Jupyter Notebook
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies (optional)
```

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Libraries Used:**

  * pandas
  * numpy
  * matplotlib / seaborn
  * scikit-learn
  * xgboost
* **IDE:** Jupyter Notebook

---

## 🔍 Project Workflow

### 1️⃣ Data Understanding

* Load dataset
* Inspect data types and structure
* Check for missing values

### 2️⃣ Exploratory Data Analysis (EDA)

* Statistical summary
* Feature distribution analysis
* Correlation heatmap

### 3️⃣ Data Preprocessing

* Feature–target split
* Train–test split
* Optional scaling

### 4️⃣ Model Building

* Decision Tree Regressor / Classifier
* XGBoost Regressor

### 5️⃣ Model Evaluation

* RMSE / MAE / SMAPE (Regression)
* Overfitting and generalization checks

### 6️⃣ Model Comparison

* Decision Tree vs XGBoost
* Bias–Variance analysis

---


### 🔍 Insights

* A fully grown Decision Tree achieved **100% accuracy** but clearly **overfit** the training data
* Limiting tree depth (`max_depth = 5`) reduced overfitting but caused **underfitting**
* Adding `min_samples_split` significantly improved generalization
* **XGBoost achieved the best performance (97%)** by combining multiple weak learners

---

## 📈 Key Results & Insights

* XGBoost outperforms a single Decision Tree due to ensemble learning
* Decision Trees are interpretable but sensitive to hyperparameters
* Alcohol and sulphates strongly influence wine quality

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Jupyter Notebook

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Run the Notebook

```bash
jupyter notebook RedWineDecisionTrees.ipynb
```

---

## 🧠 Concepts Used

This project applies several **core Machine Learning concepts** from the Andrew Ng specialization and classic tree-based learning theory:

### 🌳 Decision Tree Concepts

* **Entropy** – Measures impurity in a node; lower entropy means purer splits
* **Information Gain** – Used to select the best feature split by maximizing entropy reduction
* **Gini Impurity** – Alternative impurity measure (used internally by scikit-learn)
* **Recursive Binary Splitting** – Process of splitting nodes until stopping criteria are met
* **Stopping Criteria** – `max_depth`, `min_samples_split` to control tree growth

### ⚖️ Bias–Variance Tradeoff

* **Overfitting** – 100% training accuracy with unrestricted tree depth
* **Underfitting** – Low accuracy when tree depth is too small (`max_depth = 5`)
* **Regularization** – Controlling complexity using tree hyperparameters

### 🌲 Ensemble Learning (XGBoost)

* **Boosting** – Sequentially training weak learners to correct previous errors
* **Gradient Boosting** – Optimizing a loss function using gradient descent
* **Shrinkage (Learning Rate)** – Reduces over-correction by each tree
* **Tree Ensembles** – Combining multiple trees for better generalization

### 📊 Model Evaluation

* **Training Accuracy** – Used to diagnose overfitting
* **Error Metrics** – RMSE / MAE / SMAPE (where applicable)
* **Model Comparison** – Single tree vs ensemble model

---

## 🧠 Learnings

* Practical application of tree-based ML models
* Understanding ensemble methods
* Importance of evaluation metrics
* Structuring an ML project for GitHub

---

## 🔮 Future Enhancements

* Hyperparameter tuning (GridSearchCV)
* Cross-validation
* Feature engineering
* Model explainability using SHAP

---



