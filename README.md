# NSE F&O Direction Analyzer

Streamlit app for probability-based directional analysis of NSE F&O stocks.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- NSE F&O autocomplete/search
- Live price and recent change
- Refresh Data button
- Last Updated timestamp in IST
- 5/10/20 trading-day upside/downside probability estimates
- Trend, momentum, volatility, volume, relative strength and Nifty-regime features
- Chronological training split to reduce future-data leakage
- Explanation of positive and negative technical factors

Probabilities are statistical estimates, not guarantees. Sudden news/events can invalidate technical signals.
