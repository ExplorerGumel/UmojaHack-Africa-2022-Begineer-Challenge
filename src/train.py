import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocess import load_data, impute_missing, log_transform
from features import build_features

# ── Config ─────────────────────────────────────────────────
DATA_DIR   = "data"
# If data files live in a nested folder (e.g. downloaded archive), auto-detect it
if not (os.path.exists(os.path.join(DATA_DIR, "train.csv")) and
        os.path.exists(os.path.join(DATA_DIR, "test.csv")) and
        os.path.exists(os.path.join(DATA_DIR, "SampleSubmission.csv"))):
    for root, dirs, files in os.walk(DATA_DIR):
        if ("train.csv" in files and "test.csv" in files and
                "SampleSubmission.csv" in files):
            DATA_DIR = root
            print(f"[INFO] Found data files under: {DATA_DIR}")
            break

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
train, test, ss = load_data(
    train_path       = os.path.join(DATA_DIR, "train.csv"),
    test_path        = os.path.join(DATA_DIR, "test.csv"),
    submission_path  = os.path.join(DATA_DIR, "SampleSubmission.csv"),
)

# ── Preprocess ─────────────────────────────────────────────
train, test = impute_missing(train, test)
train, test = log_transform(train, test)

# ── Feature engineering ────────────────────────────────────
train = build_features(train)
test  = build_features(test)

# ── Split features and target ──────────────────────────────
X = train.drop("Offset_fault", axis=1)
y = train["Offset_fault"]

print(f"[OK] Features: {list(X.columns)}")
print(f"[OK] Target distribution:\n{y.value_counts(normalize=True).round(4)}")

# ── Cross-validated training ───────────────────────────────
folds            = StratifiedShuffleSplit(n_splits=10, random_state=2021)
oof_preds        = np.zeros(len(X))
test_predictions = np.zeros(len(test))
fold_scores      = []
best_clf         = None
best_fold_score  = 0.0

print("\n[TRAINING] 10-fold Stratified Shuffle Split")
print("-" * 45)

for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_trn, y_trn)

    val_preds          = clf.predict(X_val)
    val_f1             = f1_score(y_val, val_preds)
    fold_scores.append(val_f1)
    oof_preds[val_idx] = val_preds
    test_predictions  += clf.predict(test) / folds.n_splits

    if val_f1 > best_fold_score:
        best_fold_score = val_f1
        best_clf        = clf

    print(f"Fold {fold+1:02d} | Val F1: {val_f1:.4f}")

mean_f1 = float(np.mean(fold_scores))
print(f"\n[OK] Mean CV F1: {mean_f1:.4f} (+/- {np.std(fold_scores):.4f})")
print(f"[OK] Best fold F1: {best_fold_score:.4f}")

# ── Evaluation ─────────────────────────────────────────────
print("\n[EVALUATION] Out-of-fold classification report")
print(classification_report(y, oof_preds, target_names=["Normal", "Faulty"]))

# Confusion matrix
cm = confusion_matrix(y, oof_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Faulty"],
            yticklabels=["Normal", "Faulty"])
plt.title("Out-of-fold Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
print("[OK] confusion_matrix.png saved")

# Feature importance
importances = pd.Series(best_clf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(15)
plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="steelblue")
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
print("[OK] feature_importance.png saved")

# ── Save model ─────────────────────────────────────────────
model_artifact = {
    "model":         best_clf,
    "feature_names": list(X.columns),
    "mean_cv_f1":    mean_f1,
    "best_fold_f1":  best_fold_score,
}
model_path = os.path.join(OUTPUT_DIR, "gradient_boosting_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_artifact, f)
print(f"[OK] Model saved to {model_path}")

# ── Generate submission ────────────────────────────────────
ss["Offset_fault"] = [1 if x >= 0.5 else 0 for x in test_predictions]
submission_path    = os.path.join(OUTPUT_DIR, "submission.csv")
ss.to_csv(submission_path, index=False)
print(f"[OK] Submission saved to {submission_path}")
print(ss["Offset_fault"].value_counts())