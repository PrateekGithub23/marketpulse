import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
import os
import joblib


df = pd.read_csv("data/processed/ml_dataset.csv")

FEATURES = ["return_1m","return_5m","log_volume_ratio","log_trades_ratio","volatility_10m","candle_range","taker_buy_ratio"]
X = df[FEATURES]
y = df["pump"]
groups = df["pump_time"].astype(str)

splitter = GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


model = RandomForestClassifier(n_estimators=200,max_depth=6,class_weight="balanced",random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(model,"models/random_forest.pkl")

print("\nModel saved to models/random_forest.pkl")

predictions = model.predict(X_test)

print("\nClassification report:\n")

print(classification_report(y_test, predictions,digits=3))

print("Confusion matrix:")

print(confusion_matrix(y_test,predictions))



importance = pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:\n")
print(importance)