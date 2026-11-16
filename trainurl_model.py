
import pandas as pd
import joblib

from url_features import extract_url_features

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv("dataset_phishing.csv")

df = df.dropna(subset=["url", "status"])

print("Dataset shape:", df.shape)


# --------------------------------
# Extract URL Features
# --------------------------------

print("\nExtracting URL features...")

X = pd.DataFrame(
    df["url"].apply(extract_url_features).tolist()
)

y = df["status"].str.strip().str.lower().map({
    "legitimate": 0,
    "phishing": 1
})

# Remove any rows with unknown target labels
valid_rows = y.notna()

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].astype(int).reset_index(drop=True)

print("Feature shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------
# Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Define Models
# --------------------------------

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------
# Cross Validation and Evaluation
# --------------------------------

results = []
fitted_models = {}
cv_scores = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Cross-validation on training data
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1",
        n_jobs=1
    )

    cv_scores[name] = scores.mean()

    print("Mean CV F1:", round(scores.mean(), 4))

    # Fit model
    model.fit(X_train, y_train)

    fitted_models[name] = model

    # Predictions
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"]
    ))


# --------------------------------
# Model Comparison
# --------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))


# --------------------------------
# Select Best Model
# --------------------------------

best_name = max(cv_scores, key=cv_scores.get)

best_model = fitted_models[best_name]

print("\nSelected model:", best_name)


# --------------------------------
# Save Model and Feature Names
# --------------------------------

model_bundle = {
    "model": best_model,
    "feature_names": X.columns.tolist(),
    "model_name": best_name
}

joblib.dump(model_bundle, "phishguard_model.pkl")

print("\nModel saved successfully!")
