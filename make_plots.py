import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ConfusionMatrixDisplay


os.makedirs("results", exist_ok=True)

FEATURES = [
    "return_1m",
    "return_5m",
    "log_volume_ratio",
    "log_trades_ratio",
    "volatility_10m",
    "candle_range",
    "taker_buy_ratio"
]


# ---------------------------------------
# Load data + model
# ---------------------------------------

df = pd.read_csv(
    "data/processed/ml_dataset.csv"
)

model = joblib.load(
    "models/random_forest.pkl"
)

X = df[FEATURES]
y = df["pump"]

groups = df["pump_time"].astype(str)


# ---------------------------------------
# Reproduce test split
# ---------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

predictions = model.predict(X_test)


# =======================================
# 1. Confusion Matrix
# =======================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Normal", "Pump"]
)

plt.title("Pump Detection Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=200
)

plt.close()


# =======================================
# 2. Feature Importance
# =======================================

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values()

plt.figure(figsize=(9, 5))

importance.plot(
    kind="barh"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=200
)

plt.close()


# =======================================
# 3. Early Detection Lead Time
# =======================================

results = pd.read_csv(
    "data/processed/early_detection_results.csv"
)

detected = results[
    results["detected_before_peak"] == True
].copy()

labels = (
    detected["symbol"]
    + "\n"
    + detected["pump_time"].str[:10]
)

plt.figure(figsize=(10, 5))

plt.bar(
    labels,
    detected["lead_time_to_peak_minutes"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel("Minutes Before Peak")
plt.xlabel("Held-Out Pump Event")

plt.title(
    "Model Alert Lead Time Before Local Price Peak"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/early_detection_lead_time.png",
    dpi=200
)

plt.close()


print("Plots created:")
print("results/confusion_matrix.png")
print("results/feature_importance.png")
print("results/early_detection_lead_time.png")