import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit


# ==================================================
# Load dataset
# ==================================================

df = pd.read_csv(
    "data/processed/ml_dataset.csv",
    parse_dates=["datetime", "pump_time"]
)

FEATURES = [
    "return_1m",
    "return_5m",
    "log_volume_ratio",
    "log_trades_ratio",
    "volatility_10m",
    "candle_range",
    "taker_buy_ratio"
]

X = df[FEATURES]
y = df["pump"]

groups = df["pump_time"].astype(str)


# ==================================================
# Same event-level split
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()


# ==================================================
# Train model
# ==================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

model.fit(
    train[FEATURES],
    train["pump"]
)

test["prediction"] = model.predict(
    test[FEATURES]
)

test["pump_probability"] = model.predict_proba(
    test[FEATURES]
)[:, 1]


# ==================================================
# Event-level early detection
# ==================================================

results = []

for pump_time, event in test.groupby("pump_time"):

    event = event.sort_values("datetime").copy()

    pump_time = pd.Timestamp(pump_time)

    # ----------------------------------------------
    # Find price peak
    # Search from announcement through +10 minutes
    # ----------------------------------------------

    post_pump = event[
        (event["datetime"] >= pump_time) &
        (
            event["datetime"]
            <= pump_time + pd.Timedelta(minutes=10)
        )
    ]

    if len(post_pump) == 0:
        continue

    peak_index = post_pump["high"].idxmax()

    peak_time = post_pump.loc[
        peak_index,
        "datetime"
    ]

    peak_price = post_pump.loc[
        peak_index,
        "high"
    ]

    # ----------------------------------------------
    # Find first model alert
    # Only announcement → peak counts
    # ----------------------------------------------

    detection_window = event[
        (event["datetime"] >= pump_time) &
        (event["datetime"] <= peak_time)
    ]

    alerts = detection_window[
        detection_window["prediction"] == 1
    ]

    if len(alerts) > 0:

        first_alert = alerts.iloc[0]["datetime"]

        detection_delay = (
            first_alert - pump_time
        ).total_seconds() / 60

        lead_time = (
            peak_time - first_alert
        ).total_seconds() / 60

        detected = True

    else:

        first_alert = pd.NaT
        detection_delay = np.nan
        lead_time = np.nan
        detected = False

    results.append({
        "symbol": event["symbol"].iloc[0],
        "pump_time": pump_time,
        "first_alert": first_alert,
        "peak_time": peak_time,
        "peak_price": peak_price,
        "detected_before_peak": detected,
        "detection_delay_minutes": detection_delay,
        "lead_time_to_peak_minutes": lead_time
    })


results = pd.DataFrame(results)


# ==================================================
# Save
# ==================================================

results.to_csv(
    "data/processed/early_detection_results.csv",
    index=False
)


# ==================================================
# Print results
# ==================================================

print("\nEARLY DETECTION / PEAK RESULTS")
print("==============================\n")

print(results.to_string(index=False))


detected = results[
    results["detected_before_peak"]
]


print("\nSUMMARY")
print("-------")

print("Test events:", len(results))

print(
    "Detected before/at peak:",
    len(detected)
)

if len(results) > 0:

    print(
        "Detection rate:",
        round(
            len(detected) / len(results),
            3
        )
    )

if len(detected) > 0:

    print(
        "Average alert delay:",
        round(
            detected[
                "detection_delay_minutes"
            ].mean(),
            2
        ),
        "minutes"
    )

    print(
        "Median alert delay:",
        round(
            detected[
                "detection_delay_minutes"
            ].median(),
            2
        ),
        "minutes"
    )

    print(
        "Average lead time to peak:",
        round(
            detected[
                "lead_time_to_peak_minutes"
            ].mean(),
            2
        ),
        "minutes"
    )

    print(
        "Median lead time to peak:",
        round(
            detected[
                "lead_time_to_peak_minutes"
            ].median(),
            2
        ),
        "minutes"
    )