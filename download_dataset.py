import pandas as pd
import requests
import zipfile
import io
import os
import time

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


pumps = pd.read_csv("pump_telegram.csv")
pumps = pumps[pumps["exchange"].str.lower() == "binance"].copy()
pumps["date"] = pd.to_datetime(pumps["date"])
pumps = pumps.head(30)

downloaded = set()
successful = 0
failed = 0

for _, row in pumps.iterrows():

    symbol = str(row["symbol"]).upper()
    pair = f"{symbol}BTC"
    year = row["date"].year
    month = row["date"].month

    key = f"{pair}-{year}-{month:02d}"

    
    if key in downloaded:
        continue

 # add to our set
    downloaded.add(key)

    filename = f"{pair}-1m-{year}-{month:02d}.csv"
    output_path = os.path.join(RAW_DIR, filename)

    if os.path.exists(output_path):
        print("Already exists:", filename)
        successful += 1
        continue

    url = (
        f"https://data.binance.vision/data/spot/monthly/klines/"
        f"{pair}/1m/"
        f"{pair}-1m-{year}-{month:02d}.zip"
    )

    print("Downloading:", filename)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(RAW_DIR)
            print("SUCCESS")
            successful += 1

        else:
            print("NOT AVAILABLE:", response.status_code)
            failed += 1

    except Exception as e:
        print("  ERROR:", e)
        failed += 1

    time.sleep(0.2)

print("\n----------------")
print("Successful:", successful)
print("Failed:", failed)
print("----------------")