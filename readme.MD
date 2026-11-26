# MarketPulse

Early detection of cryptocurrency pump-and-dump events from one-minute market data, with a Streamlit
dashboard that replays each held-out event.

In a pump-and-dump, a Telegram group announces a coin and members buy at the same moment; price and
volume jump within a minute or two and then collapse. This project asks whether those first minutes
can be flagged from market data alone, evaluated on events the model never saw in training.

This is a feasibility study on a small test set. The results below come with their limits.

## Pipeline

```
pump_telegram.csv (known pump events, filtered to Binance)
        │
        ▼
download_dataset.py      Binance monthly 1-minute klines per coin  →  data/raw/
        │
        ▼
build_dataset.py         window per event: 60 min before → 10 min after the announcement
                         label: announcement minute + next 2 minutes = pump
                         7 features from past data only            →  data/processed/ml_dataset.csv
        │
        ▼
train_model.py           GroupShuffleSplit by event (25% of events held out)
                         RandomForest(200 trees, depth 6, balanced class weights)
        │
        ▼
early_detection.py       per held-out event: first alert vs local price peak
make_plots.py            results/*.png
app.py                   Streamlit dashboard: event replay, metrics, feature importance
```

Dataset: 36 events, 3,408 labelled minutes, 144 of them pump minutes (4.2%).

Features: 1- and 5-minute returns, log volume and trade-count ratios against a 30-minute baseline,
10-minute volatility, candle range, taker-buy ratio. Rolling baselines are shifted by one minute so a
feature never includes the minute it describes.

### Why split by event

Minutes from the same pump are nearly identical. A random row split would put neighbours of every test
minute into training and grade the model on data it has effectively seen. Here every minute of a
held-out event stays out of training.

## Results

Held-out test set: 781 minutes from 9 events, 33 of them pump minutes.

| | Original run (scikit-learn 1.0.1) | Re-run (current scikit-learn) |
|---|---|---|
| Recall | 69.7% (23 / 33) | 60.6% |
| Precision | 48.9% (23 / 47) | 52.6% |
| F1 | 0.575 | 0.563 |

Accuracy is not used: predicting "no pump" for every minute would already score 95.8%.

Event level: 7 of 9 held-out events were flagged at or before the local price peak, 2 of them with a
6–7 minute lead and 5 in the peak minute itself (`data/processed/early_detection_results.csv`).

### Limitations

- Nine held-out events from one exchange: enough to show feasibility, not to estimate real-world
  performance. The few-point difference between library versions reflects how small the test set is.
- Events come from a labelled historical dataset (2018–2019). Detecting unannounced pumps is harder and
  unmeasured.
- No trading or profit claims. An alert in the peak minute is too late to act on.

## Running it

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt

python download_dataset.py     # ~150 MB of Binance klines into data/raw/
python build_dataset.py
python train_model.py
python early_detection.py
python make_plots.py
streamlit run app.py
```

`data/processed/` and `models/` are committed, so the dashboard runs without the download step.

## Credits

The pump event list (`pump_telegram.csv`, `groups.csv`) and the research scripts `features.py`, `classifier.py` and `downloader.py` (with their
`requirements.txt`) come from
[SystemsLab-Sapienza/pump-and-dump-dataset](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset),
the material for *Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market
Manipulations* (La Morgia et al., ICCCN 2020), used under the MIT License (see `LICENSE-UPSTREAM`).

Everything else (the minute-level dataset build, features, event-grouped evaluation, early-detection
analysis and dashboard) is this project's own work. Market data from
[Binance public data](https://data.binance.vision).
