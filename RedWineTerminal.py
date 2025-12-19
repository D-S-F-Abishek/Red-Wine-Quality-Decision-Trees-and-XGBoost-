import pandas as pd 
df = pd.read_csv('./winequality-red.csv')
print(df.head())
print(df.describe())

# Diagnostic checks
print("\nColumn names:", df.columns.tolist())
print("\nDataset info:")
print(df.info())
print("\nQuality column dtype:", df['quality'].dtype)
print("\nQuality column values:")
print(df['quality'].head(20))

# Create binary classification: Good (1) or Bad (0)
df['quality_binary'] = (df['quality'] >= 6).astype(int)

print("Binary Quality Distribution:")
print(df['quality_binary'].value_counts())
print("\nDataset shape:", df.shape)
print("\nFirst 10 rows - Original vs Binary Quality:")
print(df[['quality', 'quality_binary']].head(10))

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop(['quality', 'quality_binary'], axis=1)  # Drop both original and binary quality
y = df['quality_binary']  # Use the new binary quality

# First split: 70% train, 30% temp (which will be split into CV and test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Second split: Split the 30% into 50-50 for CV (15%) and test (15%)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("X_train shape:", X_train.shape, "- 70%")
print("y_train shape:", y_train.shape)
print("\nX_cv shape:", X_cv.shape, "- 15%")
print("y_cv shape:", y_cv.shape)
print("\nX_test shape:", X_test.shape, "- 15%")
print("y_test shape:", y_test.shape)

# Verify the split
print("\nTotal samples:", len(df))
print("Train:", len(X_train), f"({len(X_train)/len(df)*100:.1f}%)")
print("CV:", len(X_cv), f"({len(X_cv)/len(df)*100:.1f}%)")
print("Test:", len(X_test), f"({len(X_test)/len(df)*100:.1f}%)")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
# print("number of classes:", len(y_train.unique()))


xgb_model.fit(X_train, y_train)

# Predictions on all three sets
y_train_pred_xgb = xgb_model.predict(X_train)
y_cv_pred_xgb = xgb_model.predict(X_cv)
y_test_pred_xgb = xgb_model.predict(X_test)

# Calculate accuracies
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb)
cv_accuracy_xgb = accuracy_score(y_cv, y_cv_pred_xgb)
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb)

print("XGBoost Model Performance:")
print("=" * 50)
print(f"Training Accuracy: {train_accuracy_xgb:.4f}")
print(f"CV Accuracy: {cv_accuracy_xgb:.4f}")
print(f"Test Accuracy: {test_accuracy_xgb:.4f}")

# Detailed evaluation on CV set
print("\nCV Classification Report:")
print(classification_report(y_cv, y_cv_pred_xgb))

# Confusion Matrix
print("\nCV Confusion Matrix:")
print(confusion_matrix(y_cv, y_cv_pred_xgb))

# # Custom Input for Prediction
# print("\n" + "=" * 50)
# print("CUSTOM WINE QUALITY PREDICTION")
# print("=" * 50)

# # Get feature names (excluding quality columns)
# feature_names = X.columns.tolist()
# print(f"\nEnter values for the following features:")
# print(f"Features: {feature_names}\n")

# # Collect user input
# user_input = []
# for feature in feature_names:
#     while True:
#         try:
#             value = float(input(f"Enter {feature}: "))
#             user_input.append(value)
#             break
#         except ValueError:
#             print(f"Invalid input. Please enter a numeric value for {feature}")

# # Convert to DataFrame for prediction
# import numpy as np
# user_df = pd.DataFrame([user_input], columns=feature_names)

# # Make prediction
# prediction = xgb_model.predict(user_df)[0]
# prediction_proba = xgb_model.predict_proba(user_df)[0]

# # Display results
# print("\n" + "=" * 50)
# print("PREDICTION RESULT")
# print("=" * 50)
# print(f"Input Features: {dict(zip(feature_names, user_input))}")
# print(f"\nPrediction: {'Good Wine (1)' if prediction == 1 else 'Bad Wine (0)'}")
# print(f"Confidence for Bad Wine (0): {prediction_proba[0]:.4f}")
# print(f"Confidence for Good Wine (1): {prediction_proba[1]:.4f}")