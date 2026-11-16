
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib


# Load dataset
df = pd.read_csv("dataset_phishing.csv")

print("Original dataset shape:", df.shape)

# Remove the URL column
# We already have numerical URL-related features
df = df.drop(columns=["url"])

# Remove constant features
constant_features = [
    "nb_or",
    "ratio_nullHyperlinks",
    "ratio_intRedirection",
    "ratio_intErrors",
    "submit_email",
    "sfh"
]

df = df.drop(columns=constant_features)

# Separate input features and target
X = df.drop(columns=["status"])
y = df["status"].map({
    "legitimate": 0,
    "phishing": 1
})

# Check the data
print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# --------------------------------
# XGBoost Baseline Model
# --------------------------------

# Create the model
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining XGBoost model...")

xgb_model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n----- XGBoost Baseline Results -----")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

#hyperparameter tuning

# --------------------------------
# Hyperparameter Tuning
# --------------------------------

# Define hyperparameter combinations
param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
    "reg_lambda": [1, 5, 10]
}

# Create XGBoost model for tuning
xgb_tuning = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb_tuning,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train using training data only
print("\nStarting Hyperparameter Tuning...")

random_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(random_search.best_score_, 4))

# Get the best model
best_model = random_search.best_estimator_

# --------------------------------
# Tuned Model Evaluation
# --------------------------------

y_pred_tuned = best_model.predict(X_test)
y_prob_tuned = best_model.predict_proba(X_test)[:, 1]

print("\n----- Tuned XGBoost Results -----")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred_tuned), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob_tuned), 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_tuned,
    target_names=["Legitimate", "Phishing"]
))

# Confusion Matrix
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_tuned,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Tuned XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# --------------------------------
# Model Comparison
# --------------------------------

baseline_accuracy = accuracy_score(y_test, y_pred)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

baseline_f1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_test, y_pred)
tuned_f1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_test, y_pred_tuned)

comparison = pd.DataFrame({
    "Model": ["Baseline XGBoost", "Tuned XGBoost"],
    "Accuracy": [baseline_accuracy, tuned_accuracy],
    "F1 Score": [baseline_f1, tuned_f1],
    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_tuned)
    ]
})

print("\n----- Model Comparison -----")
print(comparison)


#random forest

# --------------------------------
# Random Forest Baseline Model
# --------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n----- Random Forest Results -----")

print("Accuracy:",
      round(accuracy_score(y_test, rf_pred), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, rf_prob), 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    rf_pred,
    target_names=["Legitimate", "Phishing"]
))

# Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Legitimate", "Phishing"],
    yticklabels=["Legitimate", "Phishing"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# --------------------------------
# Random Forest Hyperparameter Tuning
# --------------------------------

rf_param_grid = {
    "n_estimators": [200, 300, 400, 500, 600],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced"]
}

rf_tuning_model = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)

rf_search = RandomizedSearchCV(
    estimator=rf_tuning_model,
    param_distributions=rf_param_grid,
    n_iter=20,
    scoring="f1",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("\nStarting Random Forest tuning...")

rf_search.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(rf_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(rf_search.best_score_, 4))

# Best model
best_rf_model = rf_search.best_estimator_
# Predictions
rf_tuned_pred = best_rf_model.predict(X_test)
rf_tuned_prob = best_rf_model.predict_proba(X_test)[:, 1]

print("\n----- Tuned Random Forest Results -----")

print("Accuracy:",
      round(accuracy_score(y_test, rf_tuned_pred), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, rf_tuned_prob), 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    rf_tuned_pred,
    target_names=["Legitimate", "Phishing"]
))


# --------------------------------
# Compare All Models
# --------------------------------

results = pd.DataFrame({
    "Model": [
        "Baseline XGBoost",
        "Tuned XGBoost",
        "Baseline Random Forest",
        "Tuned Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_tuned),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, rf_tuned_pred)
    ],

    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_tuned),
        precision_score(y_test, rf_pred),
        precision_score(y_test, rf_tuned_pred)
    ],

    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_tuned),
        recall_score(y_test, rf_pred),
        recall_score(y_test, rf_tuned_pred)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_tuned),
        f1_score(y_test, rf_pred),
        f1_score(y_test, rf_tuned_pred)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_tuned),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, rf_tuned_prob)
    ]
})

print("\n========== FINAL MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))

# Plot comparison
results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].plot(kind="bar", figsize=(12, 6))

plt.title("XGBoost vs Random Forest Performance")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
import joblib

joblib.dump(best_model, "phishing_xgboost_model.pkl")
joblib.dump(best_rf_model, "phishing_random_forest_model.pkl")

print("\nBoth tuned models saved successfully!")