import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- CACHED DATA FETCHING (1 hour TTL) ---
@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol, period):
    period_map = {
        "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", 
        "1 Year": "1y", "2 Years": "2y", "3 Years": "3y", 
        "4 Years": "4y", "5 Years": "5y", "Max History (IPO)": "max"
    }
    yf_period = period_map.get(period, "1y")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=yf_period)
    if df.empty or len(df) < 15:
        return None
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    # We'll compute bins later in analyze_entropy; for now just return raw data
    return df

# --- HELPER: Convert price series to USD ---
def add_close_usd(df, ticker_symbol):
    if ticker_symbol == "TSLA":
        df['Close_USD'] = df['Close']
        return df
    if ticker_symbol.endswith(".HK"):
        fx_ticker = "HKDUSD=X"
        fx = yf.Ticker(fx_ticker)
        fx_data = fx.history(start=df.index[0], end=df.index[-1] + pd.Timedelta(days=1))
        if not fx_data.empty:
            fx_series = fx_data['Close'].reindex(df.index, method='ffill')
            df['Close_USD'] = df['Close'] * fx_series
        else:
            df['Close_USD'] = df['Close'] * 0.128   # fallback
    else:
        df['Close_USD'] = df['Close']
    return df

# --- CORE ENTROPY ANALYSIS (now also returns bins) ---
def analyze_entropy(df, target_return=0.005):
    states = ['Down', 'Neutral', 'Up']
    df['State'], bins = pd.qcut(df['Log_Returns'], 3, labels=states, retbins=True)
    state_midpoints = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]

    transitions = pd.crosstab(df['State'].shift(1), df['State'], normalize='index')
    transitions = transitions.reindex(index=states, columns=states, fill_value=0)

    pi = df['State'].value_counts(normalize=True).reindex(states).values
    entropy_rate = 0
    for i, state in enumerate(states):
        row_probs = transitions.iloc[i].values
        row_probs = row_probs[row_probs > 0]
        entropy_rate += pi[i] * (-np.sum(row_probs * np.log2(row_probs)))
    predictability = (1 - (entropy_rate / np.log2(3))) * 100

    hv_ann = df['Log_Returns'].std() * np.sqrt(252)

    def obj(q):
        return np.sum(q * np.log(q + 1e-12))
    cons = [{'type': 'eq', 'fun': lambda q: np.sum(q) - 1.0},
            {'type': 'eq', 'fun': lambda q: np.sum(q * state_midpoints) - target_return}]
    res = minimize(obj, [1/3]*3, bounds=[(0, 1)]*3, constraints=cons)
    iv_ann = 0
    if res.success:
        iv_var = np.sum(res.x * (np.array(state_midpoints) - target_return)**2)
        iv_ann = np.sqrt(iv_var) * np.sqrt(252)

    return transitions, predictability, hv_ann, iv_ann, bins

# --- STREAMLIT UI ---
st.set_page_config(page_title="2026 Global EV Leaderboard", layout="wide")

company_data = {
    "BYD 🇨🇳": {"ticker": "1211.HK", "sales": "2.6M", "growth": "14%", "share": "19.9%"},
    "Geely 🇨🇳": {"ticker": "0175.HK", "sales": "1.3M", "growth": "68%", "share": "10.2%"},
    "Tesla 🇺🇸": {"ticker": "TSLA", "sales": "985K", "growth": "-11%", "share": "7.7%"},
}

st.title("⚡ Global EV Leaders: Performance & Entropy Dashboard")
st.sidebar.header("📊 Manufacturer Selection")
selected_name = st.sidebar.selectbox("Choose Company:", options=list(company_data.keys()))
info = company_data[selected_name]

st.sidebar.divider()
st.sidebar.markdown(f"**Annual Sales:** {info['sales']}")
st.sidebar.markdown(f"**YoY Growth:** {info['growth']}")
st.sidebar.markdown(f"**Market Share:** {info['share']}")
st.sidebar.divider()

# --- Stock Info with correct Google Finance format ---
st.sidebar.markdown("**🔖 Stock Info**")
st.sidebar.markdown(f"**Ticker:** `{info['ticker']}`")

# Yahoo Finance link (always works)
yahoo_url = f"https://finance.yahoo.com/quote/{info['ticker']}"
st.sidebar.markdown(f"📈 [View on Yahoo Finance]({yahoo_url})")

history_choice = st.sidebar.selectbox(
    "Analysis Window:",
    ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "Max History (IPO)"],
    index=1
)
target_ret = st.sidebar.slider("Target Daily Return (%)", -1.0, 1.0, 0.1) / 100

backtest_mode = st.sidebar.checkbox(
    "🔍 Backtest Mode (predict last day vs actual)",
    value=False,
    help="Train on all data except the most recent day, then predict that day and compare."
)

try:
    with st.spinner(f"Fetching {selected_name} data..."):
        df_full = fetch_data(info['ticker'], history_choice)

    if df_full is None:
        st.error("Insufficient data. Try a longer window or check ticker.")
        st.stop()

    df_full = add_close_usd(df_full, info['ticker'])

    if backtest_mode and len(df_full) >= 2:
        train_df = df_full.iloc[:-1].copy()
        test_row = df_full.iloc[-1].copy()

        # Analyze training set, capturing the bins
        trans, pred, hv, iv, bins = analyze_entropy(train_df, target_ret)

        # Classify test day using the training bins
        states = ['Down', 'Neutral', 'Up']
        test_state_label = pd.cut([test_row['Log_Returns']], bins=bins, labels=states, include_lowest=True)[0]
        test_row['State'] = test_state_label

        last_train_state = train_df['State'].iloc[-1]
        predicted_state = trans.loc[last_train_state].idxmax()
        pred_prob = trans.loc[last_train_state].max()
        actual_state = test_row['State']
        correct = (predicted_state == actual_state)

        # For visualization, we'll use the training bins on the full dataset?
        # Actually we want the histogram to show the distribution with the training bins.
        # So we'll create a copy of df_full with states assigned using training bins.
        df_viz = df_full.copy()
        df_viz['State'] = pd.cut(df_viz['Log_Returns'], bins=bins, labels=states, include_lowest=True)
        st.info("🧪 **Backtest Mode active** – prediction vs actual for the most recent trading day.")
    else:
        # Normal mode: analyze full dataset
        train_df = df_full.copy()
        trans, pred, hv, iv, bins = analyze_entropy(train_df, target_ret)
        last_train_state = train_df['State'].iloc[-1]
        predicted_state = trans.loc[last_train_state].idxmax()
        pred_prob = trans.loc[last_train_state].max()
        actual_state = None
        correct = None
        df_viz = df_full.copy()  # full data with states already assigned

    usd_last = df_full['Close_USD'].iloc[-1]
    local_last = df_full['Close'].iloc[-1]

    st.subheader(f"Strategy Dashboard: {selected_name} ({info['ticker']})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictability", f"{pred:.1f}%")
    c2.metric("Hist. Volatility", f"{hv:.1%}")
    c3.metric("MaxEnt Implied Vol", f"{iv:.1%}", delta=f"{iv-hv:.1%}", delta_color="inverse")
    with c4:
        st.metric("Last Price (USD)", f"{usd_last:,.2f}")
        if info['ticker'] != "TSLA":
            st.caption(f"Local: {local_last:,.2f} HKD")

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Regime Transition Matrix")
        fig = px.imshow(trans, text_auto=".2f", color_continuous_scale='Greens',
                       labels=dict(x="To State", y="From State", color="Probability"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🧠 Intelligence Summary")

        last_date = train_df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        if next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=7 - next_date.weekday())
        last_date_str = last_date.strftime('%Y-%m-%d')
        next_date_str = next_date.strftime('%Y-%m-%d')

        colors = {"Up": "#28a745", "Neutral": "#6c757d", "Down": "#dc3545"}

        st.markdown(f"""
            <div style="padding:20px; border-radius:15px; background-color:{colors[last_train_state]}; color:white; text-align:center;">
                <small>CURRENT STATE</small><br><b style="font-size:30px;">{last_train_state.upper()}</b><br>
                <small style="opacity:0.9;">Data as of: {last_date_str}</small>
            </div>
        """, unsafe_allow_html=True)

        # Enhanced card with confidence bar
        pred_card = f"""
            <div style="margin-top:10px; padding:20px; border:3px solid {colors[predicted_state]}; border-radius:15px; text-align:center; color:{colors[predicted_state]};">
                <small>NEXT LIKELY MOVE</small><br><b style="font-size:30px;">{predicted_state.upper()}</b><br>
                <small>Probability: {pred_prob:.1%}</small>
                <!--<div style="height:10px; width:100%; background-color:#e0e0e0; border-radius:5px; margin-top:10px;">
                    <div style="height:10px; width:{pred_prob*100}%; background-color:{colors[predicted_state]}; border-radius:5px;"></div>
                </div>-->
                <small style="opacity:0.9;">For: {next_date_str}</small>
            </div>
        """

        st.markdown(pred_card, unsafe_allow_html=True)

        if backtest_mode and correct is not None:
            result_color = "#28a745" if correct else "#dc3545"
            result_text = "✓ Correct" if correct else "✗ Incorrect"
            st.markdown(f"""
                <div style="margin-top:10px; padding:10px; border-radius:10px; background-color:{result_color}; color:white; text-align:center;">
                    <b>Backtest Result: {result_text}</b><br>
                    <small>Actual: {actual_state} | Predicted: {predicted_state}</small>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("### Historical Price (USD)")
    st.line_chart(df_full['Close_USD'])

    # --- NEW: Regime Distribution Histogram ---
    #st.markdown("### Daily Return Distribution with Regime Cutoffs")
    # Use df_viz which has states assigned with the correct bins (training bins in backtest mode)
    #fig_hist = go.Figure()
    #fig_hist.add_trace(go.Histogram(x=df_viz['Log_Returns'], nbinsx=50, name='Returns', marker_color='lightblue'))
    # Add vertical lines for bin edges
    #for i, edge in enumerate(bins):
    #    fig_hist.add_vline(x=edge, line_dash="dash", line_color="red",
    #                       annotation_text=f"Bin {i}", annotation_position="top right")
    #fig_hist.update_layout(xaxis_title="Log Return", yaxis_title="Frequency")
    #st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"System Error: {e}")
