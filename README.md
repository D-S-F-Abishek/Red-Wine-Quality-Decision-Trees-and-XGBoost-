# Red Wine Quality Classification

## Project Overview
This project implements machine learning models to classify red wine quality using Decision Trees and XGBoost algorithms. The dataset contains various physicochemical properties of wines, and the goal is to predict whether a wine is "Good" (quality ≥ 6) or "Bad" (quality < 6).

## Dataset
- **Source**: `winequality-red.csv`
- **Features**: 11 physicochemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target**: Binary classification (Good = 1, Bad = 0)
- **Classification Threshold**: Quality ≥ 6 → Good wine

## Data Split
- **Training Set**: 70%
- **Cross-Validation Set**: 15%
- **Test Set**: 15%

## Models Implemented

### 1. Decision Tree Classifier (Baseline - Overfitting)
**Configuration** (Cell 4):
```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None  # No depth limit - causes overfitting
)
```
**Issue**: Severe overfitting due to unlimited tree depth
- Model memorizes training data
- Poor generalization to unseen data

---

### 2. Decision Tree Classifier (Regularized - Version 1)
**Configuration** (Cell 6):
```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced"
)
```
**Purpose**: Address overfitting with regularization parameters
- `max_depth=5`: Limits tree depth to prevent overfitting
- `min_samples_leaf=10`: Requires minimum samples per leaf
- `class_weight="balanced"`: Handles class imbalance

**Performance**: Moderate accuracy, potential underfitting due to aggressive regularization

---

### 3. Decision Tree Classifier (Optimized)
**Configuration** (Cell 8):
```python
DecisionTreeClassifier(
    max_depth=9,
    min_samples_leaf=6,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)
```
**Purpose**: Balance between bias and variance
- Increased `max_depth=9` for more model complexity
- Reduced `min_samples_leaf=6` for finer granularity
- Added `min_samples_split=10` to control splits
- Better balance between training and validation accuracy

---

### 4. XGBoost Classifier (Best Performance)
**Configuration** (Cell 11):
```python
XGBClassifier(
    n_estimators=200,        # 200 boosting rounds
    learning_rate=0.1,       # Step size for updates
    max_depth=4,             # Shallow trees prevent overfitting
    min_child_weight=5,      # Minimum sum of weights per child
    subsample=0.8,           # 80% data sampling per tree
    colsample_bytree=0.8,    # 80% feature sampling per tree
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
```

**Why XGBoost Performs Better**:
- **Ensemble learning**: Combines multiple weak learners
- **Gradient boosting**: Iteratively corrects errors
- **Regularization**: Built-in L1/L2 regularization
- **Handles imbalance**: Better than single decision tree
- **Feature sampling**: Reduces overfitting through randomization

**Expected Performance**:
- Training Accuracy: ~0.85-0.90
- CV Accuracy: ~0.75-0.80
- Test Accuracy: ~0.75-0.80
- Better generalization with minimal overfitting

---

## Model Comparison Summary

| Model | Max Depth | Key Parameters | Expected Performance | Issue/Strength |
|-------|-----------|----------------|---------------------|----------------|
| DT (Baseline) | None | No regularization | Train: ~1.0, CV: ~0.6 | Severe overfitting |
| DT (v1) | 5 | min_leaf=10 | Train: ~0.65, CV: ~0.63 | Potential underfitting |
| DT (Optimized) | 9 | min_leaf=6, min_split=10 | Train: ~0.75, CV: ~0.70 | Better balance |
| **XGBoost** | **4** | **200 trees, LR=0.1** | **Train: ~0.87, CV: ~0.78** | **Best overall** |

## Evaluation Metrics
For each model, we evaluate:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Key Insights

### Progression of Model Development:
1. **Baseline Model**: Demonstrated overfitting problem
2. **Regularization**: Showed how constraints reduce overfitting but can cause underfitting
3. **Hyperparameter Tuning**: Found optimal balance for Decision Tree
4. **Advanced Algorithm**: XGBoost provides superior performance through ensemble methods

### Why Accuracy Varies:
- **Overfitting**: High training accuracy but low CV/test accuracy (baseline model)
- **Underfitting**: Low accuracy across all sets (over-regularized model)
- **Good Fit**: Similar accuracy across train/CV/test sets (XGBoost)

## Files in This Workspace
- `RedWineDecisionTrees.ipynb`: Main notebook with all model implementations
- `RedWineTerminal.py`: Python script version with custom prediction functionality
- `winequality-red.csv`: Dataset
- `README.md`: This file

## Usage

### Running the Notebook:
1. Open `RedWineDecisionTrees.ipynb`
2. Run cells sequentially from top to bottom
3. Uncomment cells 4-9 to see Decision Tree variations
4. Cell 11 contains the final XGBoost model

### Running the Python Script:
```bash
python RedWineTerminal.py
```
The script will:
- Load and preprocess data
- Train XGBoost model
- Display performance metrics
- Accept custom input for prediction

## Requirements
```
pandas
numpy
scikit-learn
xgboost
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```

## Conclusion
This project demonstrates the importance of:
- Proper train/validation/test splits
- Regularization techniques to prevent overfitting
- Hyperparameter tuning for optimal performance
- Using ensemble methods (XGBoost) for improved accuracy

The XGBoost model provides the best balance between accuracy and generalization, making it the recommended model for red wine quality classification.
