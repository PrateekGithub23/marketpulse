import requests
import zipfile
import io
import os

symbol = "BRDBTC"
year = "2018"
month = "12"
interval = "1m"

url = (
    f"https://data.binance.vision/data/spot/monthly/klines/"
    f"{symbol}/{interval}/"
    f"{symbol}-{interval}-{year}-{month}.zip"
)

print("Downloading:")
print(url)

response = requests.get(url)

print("Status:", response.status_code)

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("data/raw")

    print("Downloaded and extracted successfully.")
else:
    print("File not available.")