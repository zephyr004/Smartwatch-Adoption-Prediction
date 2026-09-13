# Smartwatch Adoption Prediction

A machine learning project that predicts whether an individual is likely to adopt a smartwatch using demographic, lifestyle, health, and technology-related factors.

## 📌 Project Overview

The goal of this project is to build a classification model that predicts **smartwatch adoption** as a binary outcome:

* `1` → Smartwatch adopted
* `0` → Smartwatch not adopted

The project focuses on understanding which factors are associated with smartwatch adoption and evaluating the performance of different machine learning approaches.

The project begins with a **baseline Logistic Regression model** and investigates the effect of removing randomly generated noise variables. The logistic regression coefficients and odds ratios are also examined to identify the strongest predictors of smartwatch adoption.

## 📊 Dataset

The dataset contains information about individuals and several variables that may influence smartwatch adoption.

The predictors include factors related to:

* Demographics
* Income and socioeconomic characteristics
* Health and lifestyle
* Technology usage
* Consumer behavior
* Geographic/accessibility factors

The target variable is:

```text
Adopt
```

where:

```text
0 = No smartwatch adoption
1 = Smartwatch adoption
```

### Random Noise Variables

The dataset also contains nine variables that were intentionally included as random noise:

```text
RV1
RV2
RV3
RV4
RV5
RV6
RV7
RV8
RV9
```

These variables are removed in the cleaned version of the dataset/model to investigate whether eliminating irrelevant predictors affects model performance.

## 🤖 Methodology

The project follows the following workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Baseline Logistic Regression
   ↓
Model Evaluation
   ↓
Remove Random Noise Variables
   ↓
Retrain Logistic Regression
   ↓
Compare Model Performance
   ↓
Analyze Coefficients & Odds Ratios
```

## 🔬 Logistic Regression

Logistic Regression is used because the target variable is binary.

The model estimates the probability that an individual will adopt a smartwatch based on the available predictors.

The baseline model is trained **without regularization**:

```python
LogisticRegression(
    penalty=None,
    max_iter=1000
)
```

This provides a simple baseline against which future regularized models can be compared.

## 📈 Model Evaluation

The model is evaluated on unseen test data using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The predicted probabilities are also used to evaluate how well the model distinguishes between adopters and non-adopters.

## 📌 Predictor Analysis

One of the main objectives of the project is to understand which variables are strong predictors of smartwatch adoption.

After fitting the logistic regression model, the coefficients are extracted:

```python
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': log_reg.coef_[0]
})
```

Odds ratios are calculated using:

```python
coefficients['Odds_Ratio'] = np.exp(
    coefficients['Coefficient']
)
```

### Interpretation

A **positive coefficient** indicates that an increase in the predictor is associated with higher odds of smartwatch adoption.

A **negative coefficient** indicates that an increase in the predictor is associated with lower odds of smartwatch adoption.

The odds ratio provides a more interpretable measure of the relationship between a predictor and adoption.

For example:

```text
Odds Ratio > 1 → higher odds of adoption
Odds Ratio < 1 → lower odds of adoption
Odds Ratio = 1 → no change in odds
```

## 🧹 Noise Variable Analysis

The project specifically investigates the effect of removing the random variables `RV1`–`RV9`.

Two models can therefore be compared:

### Model 1 — Baseline

Includes all available predictors, including the random noise variables.

### Model 2 — Clean Model

Removes:

```text
RV1–RV9
```

The performance of both models can then be compared to determine whether the random variables contribute useful predictive information or introduce unnecessary noise.

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## 📁 Project Structure

```text
Smartwatch-Adoption-Prediction/
│
├── Watch_Adoption_prediction.ipynb
├── variables.txt
├── README.md
└── dataset/
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/zephyr004/Smartwatch-Adoption-Prediction.git
```

### 2. Navigate to the project

```bash
cd Smartwatch-Adoption-Prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 4. Open the notebook

```bash
jupyter notebook Watch_Adoption_prediction.ipynb
```

Run the notebook cells sequentially to reproduce the analysis.

## 🔮 Future Work

Possible extensions to the project include:

* Applying L1 (Lasso) regularization
* Applying L2 (Ridge) regularization
* Hyperparameter tuning
* Comparing Logistic Regression with other classification algorithms
* Feature selection
* Cross-validation
* ROC curve and Precision-Recall curve analysis
* Further investigation of multicollinearity
* Comparing model performance before and after removing random noise variables

## 👤 Author

**Zephyr004**

GitHub:
https://github.com/zephyr004/Smartwatch-Adoption-Prediction

---

### 📌 Key Objective

The primary objective of this project is not only to predict smartwatch adoption, but also to understand **which variables contribute most strongly to the prediction** and whether removing irrelevant random-noise variables can improve model performance.
