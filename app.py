import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="SpikeTrade Professional Strategy", layout="wide", initial_sidebar_state="expanded")

def calc_roc(series, period):
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    shifted = series.shift(period)
    roc = np.where(np.abs(shifted) > 0.0001, 
                   (series - shifted) / np.abs(shifted) * 100, 
                   0.0)
    return pd.Series(roc, index=series.index)

def calc_z_score(series, period):
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    roc_series = calc_roc(series, 1)
    mean = roc_series.rolling(window=period).mean()
    stdev = roc_series.rolling(window=period).std()
    z_score = np.where(stdev != 0, (roc_series - mean) / stdev, 0.0)
    return pd.Series(z_score, index=series.index)

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0)
    negative_flow = money_flow.where(delta < 0, 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi

def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_bollinger_bands(close, period=20, std_dev=2):
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower

def calculate_percent_b(close, upper, lower):
    percent_b = np.where((upper - lower) != 0, 
                        (close - lower) / (upper - lower), 
                        0.5)
    return pd.Series(percent_b, index=close.index)

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def calculate_vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def calculate_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calculate_linreg(series, period, offset=0):
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i >= period - 1:
            y = series.iloc[max(0, i - period + 1):i + 1].values
            if len(y) < period:
                continue
            x = np.arange(len(y))
            if len(x) > 1 and len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                result.iloc[i] = slope * (len(y) - 1 - offset) + intercept
    return result

def interval_to_minutes(interval_str):
    if interval_str.endswith('m'):
        return int(interval_str[:-1])
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 1440
    elif interval_str.endswith('wk'):
        return int(interval_str[:-2]) * 10080
    elif interval_str.endswith('mo'):
        return int(interval_str[:-2]) * 43800
    else:
        return 1440

st.title("📈 SpikeTrade Professional Strategy")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    st.subheader("📊 Data Source")
    data_source = st.radio("Select Data Source:", ["Demo Data (AAPL)", "Custom Symbol", "Upload CSV"])
    
    if data_source == "Custom Symbol":
        symbol = st.text_input("Enter Symbol:", value="AAPL")
    else:
        symbol = "AAPL"
    
    if data_source != "Upload CSV":
        period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        interval = st.selectbox("Interval:", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
    else:
        period = "6mo"
        interval = "1d"
    
    st.markdown("---")
    
    with st.expander("🎯 Spike Detection Thresholds", expanded=False):
        spike_price_z = st.slider("Price ROC Z-Score Threshold", 0.5, 6.0, 4.0, 0.1)
        spike_rsi_z = st.slider("RSI ROC Z-Score Threshold", 0.5, 6.0, 5.0, 0.1)
        spike_obv_z = st.slider("OBV ROC Z-Score Threshold", 0.5, 6.0, 5.0, 0.1)
        spike_mfi_z = st.slider("MFI ROC Z-Score Threshold", 0.5, 6.0, 5.0, 0.1)
        spike_bb_z = st.slider("Bollinger %B ROC Z-Score Threshold", 0.5, 6.0, 5.0, 0.1)
        spike_vwap_z = st.slider("VWAP ROC Z-Score Threshold", 0.5, 6.0, 5.0, 0.1)
        spike_volume_z = st.slider("Volume ROC Z-Score Threshold", 0.5, 6.0, 6.0, 0.1)
    
    with st.expander("📈 ROC Thresholds (%)", expanded=False):
        price_roc_thresh = st.number_input("Price ROC Threshold", value=0.3, step=0.1)
        rsi_roc_thresh = st.number_input("RSI ROC Threshold", value=1.2, step=0.1)
        obv_roc_thresh = st.number_input("OBV ROC Threshold", value=2.4, step=0.1)
        mfi_roc_thresh = st.number_input("MFI ROC Threshold", value=2.4, step=0.1)
        bb_roc_thresh = st.number_input("Bollinger %B ROC Threshold", value=1.2, step=0.1)
        vwap_roc_thresh = st.number_input("VWAP ROC Threshold", value=0.6, step=0.1)
        volume_roc_thresh = st.number_input("Volume ROC Threshold", value=6.0, step=0.1)
    
    with st.expander("🕐 Indicator Periods", expanded=False):
        price_roc_period = st.number_input("Price ROC Period", value=12, min_value=1, max_value=100)
        rsi_roc_period = st.number_input("RSI ROC Period", value=4, min_value=1, max_value=100)
        obv_roc_period = st.number_input("OBV ROC Period", value=12, min_value=1, max_value=100)
        mfi_roc_period = st.number_input("MFI ROC Period", value=6, min_value=1, max_value=100)
        bb_roc_period = st.number_input("Bollinger %B ROC Period", value=6, min_value=1, max_value=100)
        vwap_roc_period = st.number_input("VWAP ROC Period", value=12, min_value=1, max_value=100)
        volume_roc_period = st.number_input("Volume ROC Period", value=12, min_value=1, max_value=100)
    
    with st.expander("💰 Risk Management", expanded=False):
        stop_loss_pct = st.slider("Stop Loss %", 0.5, 12.0, 3.0, 0.1)
        target_gain_pct = st.slider("Target Gain %", 0.5, 12.0, 2.4, 0.1)
        use_atr_stops = st.checkbox("Use ATR-Based Stops (3x ATR)", value=True)
    
    with st.expander("📊 Indicator Parameters", expanded=False):
        rsi_length = st.number_input("RSI Length", value=30, min_value=2, max_value=100)
        bb_length = st.number_input("Bollinger Bands Length", value=14, min_value=2, max_value=100)
        bb_mult = st.number_input("Bollinger Bands Std Dev", value=2.2, min_value=0.5, max_value=5.0, step=0.1)
        mfi_length = st.number_input("MFI Length", value=12, min_value=2, max_value=100)
        atr_length = st.number_input("ATR Length", value=16, min_value=2, max_value=100)
    
    with st.expander("🔮 Prediction Line Settings", expanded=False):
        pred_frequency_min = st.number_input("Prediction Frequency (Minutes)", value=45, min_value=1)
        pred_length_min = st.number_input("Prediction Horizon (Minutes)", value=45, min_value=1)
        pred_lookback_days = st.number_input("Historical Lookback (Days)", value=3, min_value=1)
        pred_sensitivity = st.slider("Prediction Sensitivity", 0.1, 5.0, 1.0, 0.1)
        max_pred_lines = st.number_input("Max Prediction Lines", value=100, min_value=1)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
elif data_source in ["Demo Data (AAPL)", "Custom Symbol"]:
    with st.spinner(f'Fetching data for {symbol}...'):
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                st.error(f"No data available for {symbol}")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
else:
    st.info("Please upload a CSV file or select a data source.")
    st.stop()

if 'Close' not in df.columns:
    st.error("CSV must contain 'Close' column")
    st.stop()

required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in df.columns:
        st.error(f"CSV must contain '{col}' column")
        st.stop()

df = df[required_cols].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']

if data_source == "Upload CSV" and len(df) > 1:
    time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
    if time_diff <= 1.5:
        current_interval = '1m'
    elif time_diff <= 7:
        current_interval = '5m'
    elif time_diff <= 20:
        current_interval = '15m'
    elif time_diff <= 45:
        current_interval = '30m'
    elif time_diff <= 90:
        current_interval = '1h'
    else:
        current_interval = '1d'
else:
    current_interval = interval if data_source != "Upload CSV" else '1d'

minutes_per_bar = interval_to_minutes(current_interval)
bars_in_day = 1440.0 / minutes_per_bar if minutes_per_bar > 0 else 1
pred_lookback_bars_calc = int(pred_lookback_days * bars_in_day)
pred_frequency_bars_calc = max(1, int(pred_frequency_min / minutes_per_bar)) if minutes_per_bar > 0 else 1
pred_length_bars_calc = max(1, int(pred_length_min / minutes_per_bar)) if minutes_per_bar > 0 else 1

with st.spinner('Calculating indicators...'):
    df['rsi'] = calculate_rsi(df['close'], int(rsi_length))
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], int(bb_length), bb_mult)
    df['percent_b'] = calculate_percent_b(df['close'], df['bb_upper'], df['bb_lower'])
    df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], int(mfi_length))
    df['obv'] = calculate_obv(df['close'], df['volume'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], int(atr_length))
    df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
    
    df['ema_9'] = calculate_ema(df['close'], 9)
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['price_roc'] = calc_roc(df['close'], int(price_roc_period))
    df['rsi_roc'] = calc_roc(df['rsi'], int(rsi_roc_period))
    df['obv_roc'] = calc_roc(df['obv'], int(obv_roc_period))
    df['mfi_roc'] = calc_roc(df['mfi'], int(mfi_roc_period))
    df['bb_roc'] = calc_roc(df['percent_b'], int(bb_roc_period))
    df['vwap_roc'] = calc_roc(df['vwap'], int(vwap_roc_period))
    df['volume_roc'] = calc_roc(df['volume'], int(volume_roc_period))
    
    df['price_z'] = calc_z_score(df['close'], int(price_roc_period))
    df['rsi_z'] = calc_z_score(df['rsi'], int(rsi_roc_period))
    df['obv_z'] = calc_z_score(df['obv'], int(obv_roc_period))
    df['mfi_z'] = calc_z_score(df['mfi'], int(mfi_roc_period))
    df['bb_z'] = calc_z_score(df['percent_b'], int(bb_roc_period))
    df['vwap_z'] = calc_z_score(df['vwap'], int(vwap_roc_period))
    df['volume_z'] = calc_z_score(df['volume'], int(volume_roc_period))
    
    df['any_spike'] = (
        (df['price_z'].fillna(0) > spike_price_z) |
        (df['rsi_z'].fillna(0) > spike_rsi_z) |
        (df['obv_z'].fillna(0) > spike_obv_z) |
        (df['mfi_z'].fillna(0) > spike_mfi_z) |
        (df['bb_z'].fillna(0) > spike_bb_z) |
        (df['vwap_z'].fillna(0) > spike_vwap_z) |
        (df['volume_z'].fillna(0) > spike_volume_z)
    )
    
    df['opposite_conditions'] = (
        (df['price_z'].fillna(0) < -spike_price_z * 0.2) |
        (df['rsi_z'].fillna(0) < -spike_rsi_z * 0.2) |
        (df['obv_z'].fillna(0) < -spike_obv_z * 0.2) |
        (df['mfi_z'].fillna(0) < -spike_mfi_z * 0.2) |
        (df['bb_z'].fillna(0) < -spike_bb_z * 0.2) |
        (df['vwap_z'].fillna(0) < -spike_vwap_z * 0.2) |
        (df['volume_z'].fillna(0) < -spike_volume_z * 0.2)
    )
    
    df['trend_filter'] = df['sma_50'] > df['sma_200']
    
    df['c_price'] = (df['price_roc'].fillna(0) > price_roc_thresh * 0.5).astype(int)
    df['c_rsi'] = (df['rsi_roc'].fillna(0) > rsi_roc_thresh * 0.5).astype(int)
    df['c_obv'] = (df['obv_roc'].fillna(0) > obv_roc_thresh * 0.5).astype(int)
    df['c_mfi'] = (df['mfi_roc'].fillna(0) > mfi_roc_thresh * 0.5).astype(int)
    df['c_bb'] = (df['bb_roc'].fillna(0) > bb_roc_thresh * 0.5).astype(int)
    df['c_vwap'] = (df['vwap_roc'].fillna(0) > vwap_roc_thresh * 0.5).astype(int)
    df['c_vol'] = (df['volume_roc'].fillna(0) > volume_roc_thresh * 0.5).astype(int)
    
    df['conditions_met'] = df['c_price'] + df['c_rsi'] + df['c_obv'] + df['c_mfi'] + df['c_bb'] + df['c_vwap'] + df['c_vol']
    df['signal_probability'] = df['conditions_met'] / 7 * 100
    
    df['buy_condition'] = df['any_spike'] & (df['conditions_met'] >= 2) & df['trend_filter']
    df['sell_condition'] = df['opposite_conditions']
    
    df['atr_stop'] = df['close'] - df['atr'] * 3.0
    df['stop_level'] = np.where(use_atr_stops, df['atr_stop'], df['close'] * (1 - stop_loss_pct / 100))
    df['target_level'] = df['close'] * (1 + target_gain_pct / 100)
    
    composite_indicator = (
        df['price_z'].fillna(0) + df['rsi_z'].fillna(0) + df['obv_z'].fillna(0) + 
        df['mfi_z'].fillna(0) + df['bb_z'].fillna(0) + df['vwap_z'].fillna(0) + 
        df['volume_z'].fillna(0)
    ) / 7.0
    df['composite_indicator'] = composite_indicator

col1, col2, col3, col4 = st.columns(4)
with col1:
    latest_price = df['close'].iloc[-1]
    st.metric("Current Price", f"${latest_price:.2f}")
with col2:
    latest_signal_prob = df['signal_probability'].iloc[-1]
    st.metric("Signal Probability", f"{latest_signal_prob:.1f}%")
with col3:
    buy_signals = df['buy_condition'].sum()
    st.metric("Total Buy Signals", buy_signals)
with col4:
    latest_trend = "Bullish" if df['trend_filter'].iloc[-1] else "Bearish"
    st.metric("Trend", latest_trend)

st.markdown("---")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=('Price Chart with Indicators', 'Signal Probability', 'Volume')
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ),
    row=1, col=1
)

fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], name='EMA 9', line=dict(color='green', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], name='EMA 20', line=dict(color='orange', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='red', width=1)), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash'), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name='BB Middle', line=dict(color='gray', width=1), showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', showlegend=True), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name='VWAP', line=dict(color='purple', width=1, dash='dot')), row=1, col=1)

buy_signals_df = df[df['buy_condition']]
if not buy_signals_df.empty:
    fig.add_trace(
        go.Scatter(
            x=buy_signals_df.index,
            y=buy_signals_df['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='lime', size=12, symbol='triangle-up')
        ),
        row=1, col=1
    )
    
    for idx in buy_signals_df.index[-5:]:
        stop = df.loc[idx, 'stop_level']
        target = df.loc[idx, 'target_level']
        fig.add_hline(y=stop, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
        fig.add_hline(y=target, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)

sell_signals_df = df[df['sell_condition']]
if not sell_signals_df.empty:
    fig.add_trace(
        go.Scatter(
            x=sell_signals_df.index,
            y=sell_signals_df['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=12, symbol='triangle-down')
        ),
        row=1, col=1
    )

lookback_bars = min(pred_lookback_bars_calc, len(df) - 1)
if lookback_bars > 0:
    df['linreg_0'] = calculate_linreg(df['composite_indicator'], lookback_bars, 0)
    df['linreg_1'] = calculate_linreg(df['composite_indicator'], lookback_bars, 1)
    
    prediction_lines_shown = 0
    bars_since_last_pred = pred_frequency_bars_calc
    
    for i in range(len(df) - 1, -1, -1):
        if i >= lookback_bars and prediction_lines_shown < int(max_pred_lines):
            if bars_since_last_pred >= pred_frequency_bars_calc:
                if pd.notna(df['linreg_0'].iloc[i]) and pd.notna(df['linreg_1'].iloc[i]):
                    y2 = df['linreg_0'].iloc[i]
                    y1 = df['linreg_1'].iloc[i]
                    slope = y2 - y1
                    
                    projected_price_change = slope * pred_length_bars_calc * df['atr'].iloc[i] * pred_sensitivity
                    projected_price = df['close'].iloc[i] + projected_price_change
                    
                    color = 'green' if slope > 0 else 'red'
                    
                    future_index = min(i + pred_length_bars_calc, len(df) - 1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[i], df.index[future_index]],
                            y=[df['close'].iloc[i], projected_price],
                            mode='lines',
                            line=dict(color=color, width=2, dash='dash'),
                            opacity=0.4,
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    prediction_lines_shown += 1
                    bars_since_last_pred = 0
            bars_since_last_pred += 1

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['signal_probability'],
        name='Signal Probability',
        line=dict(color='blue', width=2),
        fill='tozeroy'
    ),
    row=2, col=1
)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

fig.add_trace(
    go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='lightblue'
    ),
    row=3, col=1
)

fig.update_xaxes(rangeslider_visible=False)
fig.update_layout(
    height=900,
    showlegend=True,
    hovermode='x unified',
    template='plotly_dark',
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, width='stretch')

st.markdown("---")
st.subheader("📊 Current Indicator Values")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Technical Indicators**")
    st.write(f"RSI: {df['rsi'].iloc[-1]:.2f}")
    st.write(f"MFI: {df['mfi'].iloc[-1]:.2f}")
    st.write(f"ATR: {df['atr'].iloc[-1]:.4f}")
    st.write(f"OBV: {df['obv'].iloc[-1]:,.0f}")

with col2:
    st.write("**Z-Scores**")
    st.write(f"Price Z-Score: {df['price_z'].iloc[-1]:.2f}")
    st.write(f"RSI Z-Score: {df['rsi_z'].iloc[-1]:.2f}")
    st.write(f"Volume Z-Score: {df['volume_z'].iloc[-1]:.2f}")
    st.write(f"Composite: {df['composite_indicator'].iloc[-1]:.2f}")

with col3:
    st.write("**Conditions Met**")
    st.write(f"Price ROC: {'✅' if df['c_price'].iloc[-1] else '❌'}")
    st.write(f"RSI ROC: {'✅' if df['c_rsi'].iloc[-1] else '❌'}")
    st.write(f"OBV ROC: {'✅' if df['c_obv'].iloc[-1] else '❌'}")
    st.write(f"Volume ROC: {'✅' if df['c_vol'].iloc[-1] else '❌'}")

st.markdown("---")
if df['buy_condition'].iloc[-1]:
    st.success(f"🟢 BUY SIGNAL ACTIVE - Signal Probability: {df['signal_probability'].iloc[-1]:.1f}%")
    st.write(f"Stop Loss: ${df['stop_level'].iloc[-1]:.2f} | Target: ${df['target_level'].iloc[-1]:.2f}")
elif df['sell_condition'].iloc[-1]:
    st.warning(f"🔴 SELL SIGNAL ACTIVE - Opposite conditions detected")
else:
    st.info(f"⚪ NO SIGNAL - Signal Probability: {df['signal_probability'].iloc[-1]:.1f}%")

with st.expander("📋 View Raw Data"):
    st.dataframe(df.tail(50))

st.markdown("---")
st.caption("SpikeTrade Professional Strategy - Replicating TradingView PineScript Strategy | Ready for PythonAnywhere Deployment")
