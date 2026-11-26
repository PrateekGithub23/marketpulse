import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
OUTPUT = "data/processed/ml_dataset.csv"

os.makedirs("data/processed", exist_ok=True) # not make if exists

COLUMNS = ["open_time","open","high","low", "close","volume","close_time","quote_volume","number_of_trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]

pumps = pd.read_csv("pump_telegram.csv")
pumps = pumps[pumps["exchange"].str.lower() == "binance"].copy()
pumps["pump_time"] = pd.to_datetime(pumps["date"].astype(str)+ " "+ pumps["hour"].astype(str))

all_samples = []

for filename in os.listdir(RAW_DIR):

    # if not filename.endswith(".csv"):
    #     continue

    filepath = os.path.join(RAW_DIR, filename)

    # Example:
    # BRDBTC-1m-2018-12.csv
    pair = filename.split("-")[0] # gives the pair name

    symbol = pair[:-3] # sym num without BTC

    print("Processing:", filename)

    try:
        df = pd.read_csv(
            filepath,
            names=COLUMNS
        )

        if len(df) == 0:
            continue

        # Numeric conversion
        numeric_cols = ["open","high","low","close","volume","number_of_trades","taker_buy_base_volume"]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col],errors="coerce") # convert all column values to nums

     
        df["datetime"] = pd.to_datetime(df["open_time"],unit="ms")

       
        # FEATURES

        df["return_1m"] = df["close"].pct_change()
        df["return_5m"] = df["close"].pct_change(5)
        df["volume_mean_30m"] = (df["volume"].rolling(30).mean().shift(1))
        df["trades_mean_30m"] = (df["number_of_trades"].rolling(30).mean().shift(1))
        df["log_volume_ratio"] = np.log1p(df["volume"] /(df["volume_mean_30m"] + 1e-8))
        df["log_trades_ratio"] = np.log1p(df["number_of_trades"] /(df["trades_mean_30m"] + 1e-8))
        df["volatility_10m"] = (df["return_1m"].rolling(10).std().shift(1))
        df["candle_range"] = ((df["high"] - df["low"]) /(df["open"] + 1e-8))
        df["taker_buy_ratio"] = (df["taker_buy_base_volume"] /(df["volume"] + 1e-8))

        
        # Find pump events for this coin
        coin_pumps = pumps[pumps["symbol"].str.upper() == symbol]

        for _, pump in coin_pumps.iterrows():
            pump_time = pump["pump_time"]
            if not (df["datetime"].min()<= pump_time<= df["datetime"].max()):
                continue

            # Use 1 hour before and 10 min after
            start = pump_time - pd.Timedelta(hours=1)
            end = pump_time + pd.Timedelta(minutes=10)

            sample = df[(df["datetime"] >= start) &(df["datetime"] <= end)].copy()

            if len(sample) == 0:
                continue

            sample["pump"] = 0

            sample.loc[(sample["datetime"] >= pump_time) & (sample["datetime"]<= pump_time + pd.Timedelta(minutes=2)),"pump"] = 1

            sample["symbol"] = symbol
            sample["pump_time"] = pump_time

            all_samples.append(sample)

    except Exception as e:
        print("ERROR:", filename, e)


if not all_samples:
    print("No samples created.")
    raise SystemExit


dataset = pd.concat(all_samples,ignore_index=True)

FEATURES = ["return_1m","return_5m","log_volume_ratio","log_trades_ratio","volatility_10m","candle_range","taker_buy_ratio"]
final = [
    "datetime",
    "symbol",
    "pump_time",
    "pump",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades"
] + FEATURES

dataset = dataset[final]

# Convert to nan and drop them
dataset = dataset.replace([np.inf, -np.inf],np.nan)
dataset = dataset.dropna()

dataset.to_csv(OUTPUT,index=False)

print("\nDataset created!")
print("Rows:", len(dataset))
print()
print("Labels:")
print(dataset["pump"].value_counts())
print()
print("Saved to:", OUTPUT)