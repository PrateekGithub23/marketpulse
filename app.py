import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import GroupShuffleSplit


st.set_page_config(
    page_title="Market Pulse",
    page_icon="📈",
    layout="wide",
)


FEATURES = [
    "return_1m",
    "return_5m",
    "log_volume_ratio",
    "log_trades_ratio",
    "volatility_10m",
    "candle_range",
    "taker_buy_ratio",
]


@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/processed/ml_dataset.csv",
        parse_dates=["datetime", "pump_time"],
    )

    events = pd.read_csv(
        "data/processed/early_detection_results.csv",
        parse_dates=["pump_time", "first_alert", "peak_time"],
    )

    return df, events


model = load_model()
df, events = load_data()


# --------------------------------------------------
# Recreate held-out test set
# --------------------------------------------------

X = df[FEATURES]
y = df["pump"]
groups = df["pump_time"].astype(str)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)

_, test_idx = next(
    splitter.split(X, y, groups)
)

test_df = df.iloc[test_idx].copy()

test_df["prediction"] = model.predict(
    test_df[FEATURES]
)

test_df["pump_probability"] = model.predict_proba(
    test_df[FEATURES]
)[:, 1]


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📈 Market Pulse")
st.subheader("Cryptocurrency Pump-and-Dump Detection")

st.caption(
    "Interactive replay of historical pump events using "
    "Binance 1-minute data and a Random Forest classifier."
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Event Detection", "77.8%", "7 / 9 events")
c2.metric("Pump Recall", "69.7%")
c3.metric("Pump F1", "57.5%")
c4.metric("Avg Lead to Peak", "1.86 min")


# --------------------------------------------------
# Tabs
# --------------------------------------------------

overview, replay, analysis = st.tabs(
    [
        "Overview",
        "Event Replay",
        "Model Analysis",
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

with overview:

    st.header("How it works")

    st.markdown(
        """
        **Verified pump events**  
        ↓  
        **Binance 1-minute candles**  
        ↓  
        **Feature engineering**  
        ↓  
        **Random Forest classifier**  
        ↓  
        **Pump alert**  
        ↓  
        **Compare alert time with local price peak**
        """
    )

    st.header("Model signals")

    st.write(
        """
        The detector looks for sudden changes in:
        - short-term price returns
        - candle range
        - volume
        - trade count
        - volatility
        - aggressive buying
        """
    )

    st.info(
        "The strongest model features were candle range "
        "and 5-minute return."
    )

    st.header("Why event-level splitting?")

    st.write(
        """
        Entire pump events are kept together during train/test splitting.
        This prevents highly related minutes from the same event appearing
        in both training and testing.
        """
    )


# ==================================================
# EVENT REPLAY
# ==================================================

with replay:

    st.header("Historical Event Replay")

    events["label"] = (
        events["symbol"]
        + "/BTC — "
        + events["pump_time"].dt.strftime("%Y-%m-%d %H:%M")
    )

    selected = st.selectbox(
        "Choose an event",
        events["label"],
    )

    event = events[
        events["label"] == selected
    ].iloc[0]

    symbol = event["symbol"]
    pump_time = event["pump_time"]

    event_df = test_df[
        (test_df["symbol"] == symbol)
        & (test_df["pump_time"] == pump_time)
    ].sort_values("datetime")


    # Event metrics
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Asset", f"{symbol}/BTC")
    m2.metric("Pump Time", pump_time.strftime("%H:%M"))

    m3.metric(
        "First Alert",
        event["first_alert"].strftime("%H:%M")
        if pd.notna(event["first_alert"])
        else "Missed",
    )

    m4.metric(
        "Lead to Peak",
        f"{event['lead_time_to_peak_minutes']:.0f} min"
        if pd.notna(event["lead_time_to_peak_minutes"])
        else "—",
    )


    if event["detected_before_peak"]:
        st.success("Detected before or at the local price peak.")
    else:
        st.error("The event was not detected before the peak.")


    # --------------------------------------------------
    # Price chart
    # --------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=event_df["datetime"],
            y=event_df["close"],
            mode="lines",
            name="Close",
        )
    )

    fig.add_vline(
        x=pump_time,
        line_dash="dash",
        annotation_text="Pump announcement",
    )

    if pd.notna(event["first_alert"]):
        fig.add_vline(
            x=event["first_alert"],
            annotation_text="Model alert",
        )

    if pd.notna(event["peak_time"]):
        fig.add_vline(
            x=event["peak_time"],
            line_dash="dot",
            annotation_text="Peak",
        )

    fig.update_layout(
        title=f"{symbol}/BTC Price",
        height=400,
        hovermode="x unified",
        yaxis_title="Price (BTC)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


    # --------------------------------------------------
    # Probability chart
    # --------------------------------------------------

    prob_fig = go.Figure()

    prob_fig.add_trace(
        go.Scatter(
            x=event_df["datetime"],
            y=event_df["pump_probability"],
            mode="lines",
            fill="tozeroy",
            name="Pump Probability",
        )
    )

    prob_fig.add_hline(
        y=0.5,
        line_dash="dash",
        annotation_text="Detection threshold",
    )

    prob_fig.update_layout(
        title="Model Pump Probability",
        height=320,
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%",
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        prob_fig,
        width="stretch",
    )


    # --------------------------------------------------
    # Minute table
    # --------------------------------------------------

    with st.expander("View minute-level predictions"):

        st.dataframe(
            event_df[
                [
                    "datetime",
                    "close",
                    "volume",
                    "number_of_trades",
                    "pump_probability",
                    "prediction",
                ]
            ],
            hide_index=True,
            width="stretch",
        )


# ==================================================
# MODEL ANALYSIS
# ==================================================

with analysis:

    st.header("Classification Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Precision", "48.9%")
        st.metric("Recall", "69.7%")
        st.metric("F1 Score", "57.5%")

        st.info(
            "Accuracy was 95.6%, but recall and F1 are more "
            "meaningful because pump minutes are rare."
        )

    with col2:

        st.image(
            "results/confusion_matrix.png",
            caption="Confusion Matrix",
        )


    st.header("Feature Importance")

    importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    importance_fig = go.Figure(
        go.Bar(
            x=importance["Importance"],
            y=importance["Feature"],
            orientation="h",
        )
    )

    importance_fig.update_layout(
        height=400,
        xaxis_title="Importance",
    )

    st.plotly_chart(
        importance_fig,
        width="stretch",
    )


    st.header("Held-Out Events")

    st.dataframe(
        events[
            [
                "symbol",
                "pump_time",
                "first_alert",
                "peak_time",
                "detected_before_peak",
                "lead_time_to_peak_minutes",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

    st.warning(
        "This baseline detects pump onset around known historical "
        "announcement times. It does not demonstrate reliable "
        "pre-announcement prediction."
    )


st.divider()

st.caption(
    "Python · Pandas · Scikit-learn · Streamlit · Plotly"
)