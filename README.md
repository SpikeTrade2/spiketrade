# SpikeTrade Professional Strategy - Web Application

A powerful web-based trading strategy dashboard that replicates the TradingView "SpikeTrade Professional Strategy" with real-time data and fully configurable parameters.

## Features

### Real Data Integration
- **100% Real Yahoo Finance Data** - No simulated or fake data
- Supports any stock symbol (AAPL, MSFT, TSLA, etc.)
- Multiple timeframes: 1m, 5m, 15m, 30m, 1h, 1d
- Historical data periods: 1mo, 3mo, 6mo, 1y, 2y

### Technical Analysis (30+ Configurable Parameters)

#### Spike Detection Thresholds
- Price ROC Z-Score Threshold (0.5-6.0, default: 4.0)
- RSI ROC Z-Score Threshold (0.5-6.0, default: 5.0)
- OBV ROC Z-Score Threshold (0.5-6.0, default: 5.0)
- MFI ROC Z-Score Threshold (0.5-6.0, default: 5.0)
- Bollinger %B ROC Z-Score Threshold (0.5-6.0, default: 5.0)
- VWAP ROC Z-Score Threshold (0.5-6.0, default: 5.0)
- Volume ROC Z-Score Threshold (0.5-6.0, default: 6.0)

#### ROC Thresholds (%)
- Price, RSI, OBV, MFI, Bollinger %B, VWAP, and Volume ROC thresholds

#### Indicator Periods
- Configurable periods for all 7 indicators

#### Risk Management
- Stop Loss % (0.5-12.0%, default: 3.0%)
- Target Gain % (0.5-12.0%, default: 2.4%)
- ATR-Based Stops option (3x ATR)

#### Indicator Parameters
- RSI Length (default: 30)
- Bollinger Bands Length (default: 14)
- Bollinger Bands Std Dev (default: 2.2)
- MFI Length (default: 12)
- ATR Length (default: 16)

#### Prediction Line Settings
- Prediction Frequency (minutes)
- Prediction Horizon (minutes)
- Historical Lookback (days)
- Prediction Sensitivity (0.1-5.0)
- Max Prediction Lines

### Visualization
- **Interactive Candlestick Charts** with Plotly
- **EMAs**: 9, 20, 50-period exponential moving averages
- **Bollinger Bands** with customizable parameters
- **VWAP** (Volume Weighted Average Price)
- **Buy/Sell Signals** with visual markers
- **Prediction Lines** showing projected price movements
- **Stop-Loss and Target Levels** displayed on chart
- **Signal Probability Meter** (0-100%)
- **Volume Chart** synchronized with price action

### Data Options
1. **Demo Data** - Pre-loaded AAPL data
2. **Custom Symbol** - Enter any stock ticker
3. **Upload CSV** - Use your own historical data

## Installation & Running Locally

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Install Dependencies
```bash
pip install streamlit plotly pandas numpy yfinance
```

Or with uv:
```bash
uv add streamlit plotly pandas numpy yfinance
```

### Run the Application
```bash
streamlit run app.py --server.port 5000
```

The app will be available at `http://localhost:5000`

## Deployment to PythonAnywhere

### Step 1: Upload Files
1. Create a new PythonAnywhere account at https://www.pythonanywhere.com
2. Go to "Files" tab
3. Create a new directory (e.g., `spiketrade`)
4. Upload `app.py` to this directory

### Step 2: Install Dependencies
1. Open a Bash console from PythonAnywhere dashboard
2. Navigate to your app directory:
   ```bash
   cd spiketrade
   ```
3. Install required packages:
   ```bash
   pip install --user streamlit plotly pandas numpy yfinance
   ```

### Step 3: Configure Web App
1. Go to "Web" tab in PythonAnywhere
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select Python 3.11
5. In the "Code" section, set:
   - **Source code**: `/home/yourusername/spiketrade`
   - **Working directory**: `/home/yourusername/spiketrade`

### Step 4: Configure WSGI File
Edit the WSGI configuration file (`/var/www/yourusername_pythonanywhere_com_wsgi.py`):

```python
import sys
import os

path = '/home/yourusername/spiketrade'
if path not in sys.path:
    sys.path.append(path)

os.chdir(path)

from streamlit.web import cli as stcli
import sys

def application(environ, start_response):
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
    stcli.main()
```

### Step 5: Reload and Access
1. Click "Reload" button in the Web tab
2. Your app will be available at: `https://yourusername.pythonanywhere.com`

## CSV Upload Format

If uploading custom data, use this CSV format:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01 09:30:00,150.00,151.50,149.50,151.00,1000000
2024-01-01 09:35:00,151.00,152.00,150.75,151.75,950000
...
```

**Required Columns**: Date, Open, High, Low, Close, Volume

## Technical Indicators Calculated

The application calculates the following indicators matching the TradingView PineScript strategy:

1. **RSI** - Relative Strength Index (Wilder smoothing)
2. **MFI** - Money Flow Index
3. **OBV** - On-Balance Volume
4. **Bollinger Bands** - Upper, Middle, Lower bands
5. **%B** - Bollinger Bands percent B
6. **VWAP** - Volume Weighted Average Price
7. **ATR** - Average True Range (Wilder smoothing)
8. **EMAs** - 9, 20, 50 period exponential moving averages
9. **ROC** - Rate of Change for all 7 indicators
10. **Z-Scores** - Statistical z-scores for spike detection

## Strategy Logic

The strategy implements the following logic:

1. **Spike Detection**: Monitors z-scores of 7 indicators for unusual movements
2. **ROC Confirmation**: Validates spikes with rate-of-change thresholds
3. **Trend Filter**: Uses SMA 50/200 crossover for trend direction
4. **Signal Generation**: Buy when spike detected + ≥2 ROC conditions met + bullish trend
5. **Risk Management**: Automatic stop-loss and target calculation
6. **Prediction Lines**: Linear regression-based price projections

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── README.md             # This file
├── pyproject.toml        # Python dependencies
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Performance Notes

- Data fetching may take 2-5 seconds depending on timeframe
- Larger datasets (1m interval, 2y period) require more processing time
- Prediction line calculations are optimized for performance

## Support & Issues

This application replicates the TradingView SpikeTrade Professional Strategy with full parameter configurability. All settings from the original PineScript are available and functional.

## License

MIT License - Free to use and modify

## Disclaimer

This tool is for educational and research purposes only. Not financial advice. Always do your own research before making investment decisions.
