# SpikeTrade Professional Strategy - Trading Dashboard

## Project Overview

A professional-grade web application that replicates the TradingView "SpikeTrade Professional Strategy" using Streamlit and real-time Yahoo Finance data. The application provides a complete trading strategy dashboard with all original PineScript parameters fully configurable through an intuitive web interface.

## Current Status

✅ **COMPLETED** - All features implemented and tested
- 100% real Yahoo Finance data integration (no simulated/fake data)
- 30+ configurable parameters matching PineScript strategy
- Interactive Plotly candlestick charts with technical indicators
- Buy/sell signal detection with visual markers
- Prediction lines using linear regression
- Signal probability meter and risk management tools
- CSV upload capability for custom data
- Ready for deployment to PythonAnywhere

## Key Features

### Data Integration
- **Real-time data** from Yahoo Finance API
- Support for any stock symbol (AAPL, MSFT, TSLA, etc.)
- Multiple intervals: 1m, 5m, 15m, 30m, 1h, 1d
- Historical periods: 1mo to 2y
- CSV upload for custom datasets

### Technical Analysis (All Configurable)

**Spike Detection Thresholds (7 indicators):**
- Price, RSI, OBV, MFI, Bollinger %B, VWAP, Volume Z-Score thresholds

**ROC Thresholds (7 indicators):**
- Rate of Change thresholds for each indicator

**Indicator Periods:**
- Configurable periods for all calculations

**Risk Management:**
- Stop Loss % (0.5-12%)
- Target Gain % (0.5-12%)
- ATR-based stops option (3x ATR)

**Indicator Parameters:**
- RSI Length (Wilder smoothing)
- Bollinger Bands (length + std dev)
- MFI Length
- ATR Length (Wilder smoothing)

**Prediction Settings:**
- Frequency, horizon, lookback, sensitivity
- Linear regression-based price projections

### Visualization
- Interactive candlestick charts
- EMAs: 9, 20, 50
- Bollinger Bands overlays
- VWAP line
- Buy/sell signal markers
- Prediction lines (green=bullish, red=bearish)
- Stop-loss and target levels
- Signal probability chart
- Volume chart

## Technical Implementation

### Architecture
- **Framework**: Streamlit
- **Charting**: Plotly
- **Data**: Yahoo Finance (yfinance)
- **Calculations**: Pandas + NumPy
- **Deployment**: Port 5000 (webview)

### Key Technical Decisions

1. **Wilder Smoothing**: RSI and ATR use exponential weighted moving averages matching PineScript's ta.rsi() and ta.atr()

2. **Prediction Lines**: 
   - Interval-aware calculation (converts minutes to bars)
   - Linear regression on composite indicator
   - Frequency-throttled drawing
   - Matches PineScript timeframe logic

3. **ROC and Z-Scores**:
   - Custom implementations matching PineScript calc_roc() and calc_z_score()
   - Handles edge cases (division by zero, null values)

4. **Signal Logic**:
   - Spike detection across 7 indicators
   - ROC confirmation (≥2 conditions)
   - Trend filter (SMA 50/200)
   - Opposite conditions for sell signals

### File Structure
```
.
├── app.py                          # Main application
├── README.md                       # User documentation
├── PYTHONANYWHERE_DEPLOYMENT.md   # Deployment guide
├── replit.md                      # This file
├── pyproject.toml                 # Dependencies
├── .streamlit/
│   └── config.toml               # Streamlit config
└── attached_assets/
    └── [PineScript reference]     # Original strategy code
```

## Dependencies

```toml
streamlit >= 1.51.0
plotly >= 6.4.0
pandas
numpy
yfinance >= 0.2.66
```

## Running the Application

### Local Development
```bash
streamlit run app.py --server.port 5000
```

### Replit
The app is configured to run automatically. Click "Run" or use the webview.

## Deployment to PythonAnywhere

See `PYTHONANYWHERE_DEPLOYMENT.md` for complete step-by-step instructions.

**Quick summary:**
1. Upload app.py to PythonAnywhere
2. Install dependencies: `pip install streamlit plotly pandas numpy yfinance`
3. Configure WSGI file for Streamlit
4. Create .streamlit/config.toml
5. Reload web app
6. Access at yourusername.pythonanywhere.com

## Recent Changes

### November 17, 2024

**Implemented:**
- Complete SpikeTrade strategy replication
- All 30+ parameters from PineScript
- Real Yahoo Finance data integration
- Interactive Plotly charts with all indicators
- ROC and Z-Score calculations
- Buy/sell signal detection
- Prediction lines with linear regression
- Signal probability meter
- Risk management (stop-loss, targets)
- CSV upload capability

**Bug Fixes:**
- Fixed Wilder smoothing for RSI and ATR (was using simple rolling mean)
- Fixed prediction line frequency gating (now honors pred_frequency_min)
- Fixed interval-to-bars conversion for prediction calculations
- Added CSV interval auto-detection
- Fixed unbound variable warnings for period/interval

**Testing:**
- End-to-end tests passed ✅
- All UI elements verified ✅
- Data source switching tested (AAPL → MSFT) ✅
- Chart rendering confirmed ✅
- All 30+ parameters functional ✅

## User Preferences

**Data Display:**
- User prefers REAL data (Yahoo Finance) over simulated data
- Default symbol: AAPL
- Default interval: 1d
- Default period: 6mo

**UI Preferences:**
- Dark theme (Plotly dark template)
- Sidebar for all controls
- Collapsible parameter sections
- Metric cards at top
- Large interactive chart

## Strategy Logic Summary

The SpikeTrade strategy identifies trading opportunities through:

1. **Spike Detection**: Monitors Z-scores of 7 indicators for unusual movements above configured thresholds

2. **ROC Confirmation**: Validates spikes with rate-of-change checks (requires ≥2/7 conditions met)

3. **Trend Filter**: Uses SMA 50/200 crossover to ensure trades align with overall trend

4. **Buy Signals**: Generated when spike detected + sufficient ROC conditions + bullish trend

5. **Sell Signals**: Triggered by opposite conditions (negative spikes)

6. **Risk Management**: Automatic stop-loss and target calculation based on ATR or percentage

7. **Predictions**: Linear regression on composite indicator for price projections

## Performance Notes

- Data fetching: 2-5 seconds (depends on timeframe)
- Chart rendering: < 1 second
- Indicator calculations: < 1 second
- Total load time: 3-7 seconds typical

**Optimization tips:**
- Use larger intervals (1h, 1d) for faster loading
- Limit prediction lines to 20-50 for better performance
- Shorter time periods load faster than longer ones

## Known Limitations

1. **Data Source**: Limited to Yahoo Finance availability
   - Some symbols may not have intraday data
   - Rate limiting possible with excessive requests

2. **Free PythonAnywhere**: 
   - Slower performance than paid tiers
   - Auto-sleep after inactivity
   - Limited CPU time

3. **Browser**: 
   - Large datasets may be slow on older browsers
   - Mobile experience not optimized

## Future Enhancements (Not Implemented)

Potential improvements for future versions:
- Live data updates (websocket streaming)
- Backtesting engine with performance metrics
- Multiple symbol comparison
- Alert system (email/webhook)
- Export trade history to CSV
- Custom indicator builder
- Multi-timeframe analysis
- Mobile-responsive UI improvements

## Support & Maintenance

**Testing**: 
- Run end-to-end tests regularly with different symbols
- Verify Yahoo Finance API connectivity
- Check for package updates

**Deployment**:
- Restart workflow after code changes
- Monitor error logs in PythonAnywhere
- Keep dependencies updated

**Documentation**:
- README.md for users
- PYTHONANYWHERE_DEPLOYMENT.md for deployment
- This file (replit.md) for development notes

## Credits

- Original Strategy: SpikeTrade Professional Strategy (TradingView PineScript)
- Implementation: Streamlit + Plotly + Yahoo Finance
- Framework: Replit for development
- Deployment: PythonAnywhere compatible
