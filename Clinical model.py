import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("kidney-stone-dataset.csv")

print(df.head())
print(df.isnull().sum())
df.fillna(df.median(), inplace=True)

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


X = df.drop("target", axis=1)
y = df["target"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42
)

svm_model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Kidney Stone Prediction – Clinical Data")
plt.show()

background = X_train[:50]

explainer = shap.KernelExplainer(
    svm_model.predict_proba,
    background
)

# Explain a few test samples
shap_values = explainer.shap_values(X_test[:20])

# ---- IMPORTANT PART ----
# Compute mean absolute SHAP values for class 1 (stone present)
shap_importance = np.mean(np.abs(shap_values[1]), axis=0)

# Bar plot (SAFE + STABLE)
shap.bar_plot(
    shap_importance,
    feature_names=X.columns
)


def estimate_severity(row):
    score = 0

    # High urinary calcium → stone formation
    if row["calc"] > 10:
        score += 2

    # Low urine pH → uric acid stones
    if row["ph"] < 5.5:
        score += 1

    # High osmolality → dehydration / concentrated urine
    if row["osmo"] > 800:
        score += 2

    # High specific gravity → concentrated urine
    if row["gravity"] > 1.020:
        score += 1

    if score <= 2:
        return "LOW"
    elif score <= 4:
        return "MEDIUM"
    else:
        return "HIGH"


def natural_suggestions(severity):
    if severity == "LOW":
        return [
            "Increase daily water intake (2.5–3L)",
            "Reduce salt consumption",
            "Eat citrus fruits"
        ]
    elif severity == "MEDIUM":
        return [
            "Avoid oxalate-rich foods (spinach, nuts)",
            "Drink lemon water",
            "Moderate animal protein"
        ]
    else:
        return [
            "Strict hydration (>3L/day)",
            "Limit calcium supplements",
            "Medical consultation advised"
        ]

sample = df.iloc[0]
severity = estimate_severity(sample)
suggestions = natural_suggestions(severity)

print("Prediction:", "Stone Present" if y_pred[0] == 1 else "No Stone")
print("Severity:", severity)
print("Suggestions:", suggestions)

import joblib

joblib.dump(svm_model, "clinical_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")


