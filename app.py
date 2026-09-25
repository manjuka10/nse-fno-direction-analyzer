import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title='NSE F&O Direction Analyzer', page_icon='📈', layout='wide')
IST = ZoneInfo('Asia/Kolkata')
HORIZONS = [5, 10, 20]
# Weighted multi-horizon probability: slightly more weight to the 20-day signal,
# while retaining meaningful short/medium-term information.
HORIZON_WEIGHTS = {5: 0.30, 10: 0.30, 20: 0.40}


@st.cache_data(ttl=3600, show_spinner=False)
def fno_universe():
    """Fetch the NSE F&O individual-security universe dynamically.

    No hardcoded stock list is used. NSE's live API is the primary source.
    The official NSE underlyings page is a dynamic fallback.
    """
    index_symbols = {
        'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50',
        'NIFTYIT', 'NIFTYFINSERVICE', 'NIFTYINFRA', 'NIFTYCPSE', 'NIFTYPSE'
    }
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&.')

    def clean(values):
        out = []
        for value in values:
            if not isinstance(value, str):
                continue
            symbol = value.strip().upper()
            if not (2 <= len(symbol) <= 25) or symbol in index_symbols:
                continue
            if all(ch in allowed for ch in symbol):
                out.append(symbol)
        return sorted(set(out))

    # Primary: NSE live API.
    try:
        import requests
        s = requests.Session()
        s.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json,text/plain,*/*',
            'Referer': 'https://www.nseindia.com/'
        })
        s.get('https://www.nseindia.com', timeout=10)
        r = s.get('https://www.nseindia.com/api/underlying-information', timeout=20)
        if r.ok:
            vals = []

            def walk(x):
                if isinstance(x, list):
                    for v in x:
                        walk(v)
                elif isinstance(x, dict):
                    if isinstance(x.get('symbol'), str):
                        vals.append(x['symbol'])
                    for v in x.values():
                        walk(v)

            walk(r.json())
            vals = clean(vals)
            if len(vals) >= 100:
                return vals
    except Exception:
        pass

    # Dynamic fallback: official NSE page; no manual stock list.
    try:
        import requests
        from io import StringIO
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'})
        r = s.get(
            'https://www.nseindia.com/static/products-services/equity-derivatives-list-underlyings-information',
            timeout=20,
        )
        if r.ok:
            tables = pd.read_html(StringIO(r.text))
            for table in tables:
                cols = {str(c).strip().upper(): c for c in table.columns}
                symbol_col = next((c for k, c in cols.items() if 'SYMBOL' in k), None)
                if symbol_col is not None:
                    vals = clean(table[symbol_col].tolist())
                    if len(vals) >= 100:
                        return vals
    except Exception:
        pass

    return []


@st.cache_data(ttl=900, show_spinner=False)
def history(symbol):
    try:
        d = yf.Ticker(symbol + '.NS').history(
            period='10y', interval='1d', auto_adjust=False, actions=False
        )
        if d.empty:
            return pd.DataFrame()
        d = d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def live(symbol):
    try:
        t = yf.Ticker(symbol + '.NS')
        d = t.history(period='2d', interval='5m', auto_adjust=False, prepost=False, actions=False)
        q = d['Close'].dropna()
        if q.empty:
            return None, None, None
        price = float(q.iloc[-1])
        daily = t.history(period='10d', interval='1d', auto_adjust=False, actions=False)['Close'].dropna()
        if len(daily) >= 2:
            prev = float(daily.iloc[-1]) if daily.index[-1].date() != datetime.now(IST).date() else float(daily.iloc[-2])
        else:
            prev = float(q.iloc[0])
        return price, price - prev, (price / prev - 1) * 100
    except Exception:
        return None, None, None


@st.cache_data(ttl=900, show_spinner=False)
def nifty():
    try:
        d = yf.Ticker('^NSEI').history(
            period='10y', interval='1d', auto_adjust=False, actions=False
        )
        d = d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d
    except Exception:
        return pd.DataFrame()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    a = up.ewm(alpha=1 / n, adjust=False).mean()
    b = dn.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + a / b.replace(0, np.nan))


def make_features(d, market):
    x = d.copy()
    for k in [10, 21, 50, 100, 200]:
        x[f'E{k}'] = x.Close.ewm(span=k, adjust=False).mean()
        x[f'S{k}'] = x[f'E{k}'].pct_change(10) * 100

    tr = pd.concat([
        x.High - x.Low,
        (x.High - x.Close.shift()).abs(),
        (x.Low - x.Close.shift()).abs(),
    ], axis=1).max(axis=1)
    x['ATR'] = tr.rolling(14).mean()
    x['ATRp'] = x.ATR / x.Close * 100
    x['RSI'] = rsi(x.Close)

    up = x.High.diff()
    dn = -x.Low.diff()
    p = up.where((up > dn) & (up > 0), 0)
    m = dn.where((dn > up) & (dn > 0), 0)
    a = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * p.ewm(alpha=1 / 14, adjust=False).mean() / a
    mdi = 100 * m.ewm(alpha=1 / 14, adjust=False).mean() / a
    x['ADX'] = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    e12 = x.Close.ewm(span=12, adjust=False).mean()
    e26 = x.Close.ewm(span=26, adjust=False).mean()
    mac = e12 - e26
    x['MACD'] = mac - mac.ewm(span=9, adjust=False).mean()

    for k in [1, 5, 10, 21, 63, 126]:
        x[f'R{k}'] = x.Close.pct_change(k) * 100

    x['V20'] = x.Close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    x['V60'] = x.Close.pct_change().rolling(60).std() * np.sqrt(252) * 100
    x['VR'] = x.Volume / x.Volume.rolling(20).mean()
    x['D21'] = (x.Close - x.E21) / x.E21 * 100
    x['D50'] = (x.Close - x.E50) / x.E50 * 100
    x['D200'] = (x.Close - x.E200) / x.E200 * 100
    x['HH20'] = x.High.rolling(20).max()
    x['LL20'] = x.Low.rolling(20).min()
    x['BO'] = (x.Close > x.HH20.shift()).astype(int)
    x['BD'] = (x.Close < x.LL20.shift()).astype(int)

    market_close = market[['Close']].rename(columns={'Close': 'NC'}).reindex(x.index).ffill()
    x = x.join(market_close, how='left')
    x['RS1'] = (x['R21'] - x['NC'].pct_change(21) * 100)
    x['RS3'] = (x['R63'] - x['NC'].pct_change(63) * 100)
    x['RS6'] = (x['R126'] - x['NC'].pct_change(126) * 100)
    n50 = x['NC'].ewm(span=50, adjust=False).mean()
    n200 = x['NC'].ewm(span=200, adjust=False).mean()
    x['NRET'] = x['NC'].pct_change(21) * 100
    x['NV'] = x['NC'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    x['NT'] = (x['NC'] > n50).astype(int)
    x['NLT'] = (x['NC'] > n200).astype(int)
    return x.replace([np.inf, -np.inf], np.nan)


FEATS = [
    'R1', 'R5', 'R10', 'R21', 'R63', 'R126', 'RSI', 'ADX', 'MACD',
    'V20', 'V60', 'VR', 'ATRp', 'D21', 'D50', 'D200', 'S10', 'S21',
    'S50', 'S100', 'S200', 'BO', 'BD', 'RS1', 'RS3', 'RS6', 'NRET',
    'NV', 'NT', 'NLT'
]


def live_adjusted_row(x, price):
    """Create a current-state feature row using the live price.

    Historical observations remain based on completed daily closes. For the
    current prediction only, today's close-dependent features are updated with
    the latest live price so a large intraday move can affect the signal.
    """
    if price is None or not np.isfinite(price):
        return x[FEATS].iloc[[-1]].copy()

    work = x[['Close', 'E10', 'E21', 'E50', 'E100', 'E200', 'NC']].copy()
    work.loc[work.index[-1], 'Close'] = float(price)

    # Rebuild a compact set of price-derived current features recursively.
    out = x.iloc[-1].copy()
    prev = float(x.Close.iloc[-2])
    e = {}
    for k in [10, 21, 50, 100, 200]:
        alpha = 2 / (k + 1)
        e[k] = alpha * float(price) + (1 - alpha) * float(x[f'E{k}'].iloc[-2])
        out[f'E{k}'] = e[k]
        out[f'D{k}'] = (price - e[k]) / e[k] * 100
        out[f'S{k}'] = (e[k] / float(x[f'E{k}'].iloc[-11]) - 1) * 100 if len(x) >= 11 else out[f'S{k}']

    out['R1'] = (price / prev - 1) * 100
    for k in [5, 10, 21, 63, 126]:
        if len(x) > k:
            out[f'R{k}'] = (price / float(x.Close.iloc[-1-k]) - 1) * 100

    # Update RSI using the live close as the newest observation.
    close_series = x.Close.copy()
    close_series.iloc[-1] = price
    out['RSI'] = float(rsi(close_series).iloc[-1])

    # Approximate MACD with the current live close replacing the latest close.
    e12_prev = float((x.Close.ewm(span=12, adjust=False).mean()).iloc[-2])
    e26_prev = float((x.Close.ewm(span=26, adjust=False).mean()).iloc[-2])
    a12 = 2 / 13
    a26 = 2 / 27
    e12_now = a12 * price + (1 - a12) * e12_prev
    e26_now = a26 * price + (1 - a26) * e26_prev
    mac_now = e12_now - e26_now
    mac_series = x.Close.ewm(span=12, adjust=False).mean() - x.Close.ewm(span=26, adjust=False).mean()
    mac_signal_prev = float(mac_series.ewm(span=9, adjust=False).mean().iloc[-2])
    sig_now = (2 / 10) * mac_now + (1 - 2 / 10) * mac_signal_prev
    out['MACD'] = mac_now - sig_now

    # Relative strength and market-regime features.
    out['RS1'] = out['R21'] - float(x.NC.pct_change(21).iloc[-1] * 100)
    out['RS3'] = out['R63'] - float(x.NC.pct_change(63).iloc[-1] * 100)
    out['RS6'] = out['R126'] - float(x.NC.pct_change(126).iloc[-1] * 100)

    # Breakout/breakdown flags use completed prior-day levels to avoid looking ahead.
    out['BO'] = int(price > float(x.HH20.iloc[-2])) if pd.notna(x.HH20.iloc[-2]) else out['BO']
    out['BD'] = int(price < float(x.LL20.iloc[-2])) if pd.notna(x.LL20.iloc[-2]) else out['BD']
    return pd.DataFrame([out])[FEATS]


def train_and_predict(x, h, current_row):
    """Train with only observations having a known h-day future outcome.

    A chronological train/validation/test design is used. The model is first
    fit on the earliest 70%, then calibrated on the following 15% using a
    logistic calibration layer. A fresh model is then fit on the first 85%
    and evaluated on the final 15% for the displayed historical metrics.
    Finally, a model is fit on all currently eligible historical observations
    and the calibrated probability is applied to the current state.
    """
    future = x.Close.shift(-h)
    d = x[FEATS].copy()
    ok = d.notna().all(axis=1) & future.notna()
    d = d.loc[ok]
    y = (future.loc[ok] > x.Close.loc[ok]).astype(int)

    if len(d) < 700 or y.nunique() < 2:
        return None

    n = len(d)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end < 400 or val_end <= train_end or n - val_end < 60:
        return None

    base_params = dict(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=3.0,
        random_state=42,
    )

    # Stage 1: model for out-of-sample calibration data.
    base1 = make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(**base_params))
    base1.fit(d.iloc[:train_end], y.iloc[:train_end])
    val_prob = base1.predict_proba(d.iloc[train_end:val_end])[:, 1]

    calibrator = LogisticRegression(C=1.0, solver='lbfgs')
    calibrator.fit(val_prob.reshape(-1, 1), y.iloc[train_end:val_end])

    # Stage 2: model trained on all data before the final test segment.
    base_test = make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(**base_params))
    base_test.fit(d.iloc[:val_end], y.iloc[:val_end])
    test_raw = base_test.predict_proba(d.iloc[val_end:])[:, 1]
    test_prob = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_y = y.iloc[val_end:].to_numpy()
    accuracy = float(accuracy_score(test_y, test_pred))
    brier = float(brier_score_loss(test_y, test_prob))

    # Final model uses all known historical observations. The calibration
    # layer is kept from strictly out-of-sample historical predictions.
    final_model = make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(**base_params))
    final_model.fit(d, y)
    raw_current = float(final_model.predict_proba(current_row)[0, 1])
    calibrated_current = float(calibrator.predict_proba(np.array([[raw_current]]))[0, 1])

    # Prevent an unstable one-model extreme from being presented as certainty.
    # This is a display/model-risk guard, not a claim that the true probability
    # is bounded to this range.
    calibrated_current = float(np.clip(calibrated_current, 0.05, 0.95))

    return {
        'prob': calibrated_current,
        'samples': int(n),
        'test_samples': int(len(test_y)),
        'accuracy': accuracy,
        'brier': brier,
        'base_rate': float(y.iloc[:val_end].mean()),
    }


def confidence_label(res):
    """Translate holdout quality into a conservative reliability label.

    This is a model-reliability label, not a probability of the trade outcome.
    Brier skill is measured against a constant base-rate forecast on the same
    holdout sample, so the label is relative to a simple benchmark.
    """
    base = float(res.get('base_rate', 0.5))
    brier = float(res.get('brier', 0.25))
    acc = float(res.get('accuracy', 0.5))
    samples = int(res.get('test_samples', 0))
    baseline_brier = max(base * (1.0 - base), 1e-9)
    skill = 1.0 - brier / baseline_brier
    majority = max(base, 1.0 - base)
    lift = acc - majority

    if samples >= 150 and skill >= 0.10 and lift >= 0.03:
        return 'High'
    if samples >= 100 and skill >= 0.03 and lift >= 0.01:
        return 'Moderate'
    return 'Low'


def technical_bias(x, price=None):
    """Compact technical-bias summary, kept separate from the ML direction."""
    r = x.iloc[-1].copy()
    if price is not None and np.isfinite(price):
        r.Close = float(price)
        r.D21 = (float(price) - float(r.E21)) / float(r.E21) * 100

    score = 0
    if r.Close > r.E21 > r.E50 > r.E200:
        score += 2
    elif r.Close < r.E21 < r.E50 < r.E200:
        score -= 2
    else:
        score += 0

    if r.S21 > 0 and r.S50 > 0:
        score += 1
    elif r.S21 < 0 and r.S50 < 0:
        score -= 1

    if r.RS1 > 0 and r.RS3 > 0:
        score += 1
    elif r.RS1 < 0 and r.RS3 < 0:
        score -= 1

    if r.R21 > 0 and r.R63 > 0:
        score += 1
    elif r.R21 < 0 and r.R63 < 0:
        score -= 1

    if 2 <= r.D21 <= 7:
        score += 1
    elif r.D21 < -5 or r.D21 > 10:
        score -= 1

    if r.ADX >= 20:
        score += 1 if r.R21 > 0 else -1 if r.R21 < 0 else 0

    if r.VR >= 1:
        score += 1 if r.R1 > 0 else -1 if r.R1 < 0 else 0

    if r.BO:
        score += 1
    if r.BD:
        score -= 1

    if r.NT:
        score += 1
    else:
        score -= 1

    if score >= 3:
        return 'Bullish', score
    if score <= -3:
        return 'Bearish', score
    return 'Neutral', score


def context(x, price=None):
    r = x.iloc[-1].copy()
    if price is not None:
        r.Close = price
        r.D21 = (price - r.E21) / r.E21 * 100
    pos, neg = [], []
    if r.Close > r.E21 > r.E50 > r.E200:
        pos.append('Bullish EMA structure')
    elif r.Close < r.E21 < r.E50 < r.E200:
        neg.append('Bearish EMA structure')
    else:
        neg.append('EMA structure is mixed')
    if r.S21 > 0 and r.S50 > 0:
        pos.append('21/50 EMA rising')
    else:
        neg.append('21/50 EMA slope is weak')
    if r.RS1 > 0 and r.RS3 > 0:
        pos.append('Outperforming Nifty over 1M and 3M')
    else:
        neg.append('Relative strength is mixed')
    if r.R21 > 0 and r.R63 > 0:
        pos.append('Positive 1M and 3M momentum')
    else:
        neg.append('Momentum is mixed')
    if 2 <= r.D21 <= 7:
        pos.append('Moderate 21 EMA distance')
    elif r.D21 > 7:
        neg.append('Price is extended above 21 EMA')
    elif r.D21 < -5:
        neg.append('Price is materially below 21 EMA')
    if r.ADX >= 20:
        pos.append('Trend strength is meaningful')
    else:
        neg.append('Trend strength is modest')
    if r.VR >= 1:
        pos.append('Volume above 20-day average')
    else:
        neg.append('Volume below 20-day average')
    if r.BO:
        pos.append('Recent 20-day breakout')
    if r.BD:
        neg.append('Recent 20-day breakdown')
    if r.NT:
        pos.append('Nifty above 50 EMA')
    else:
        neg.append('Nifty below 50 EMA')
    return pos, neg


st.title('📈 NSE F&O Stock Direction Analyzer')
st.caption('Probability-based directional analysis using historical out-of-sample calibration and current market state. Probabilities are estimates, not guarantees.')

c1, c2, c3 = st.columns([1.2, 2.5, 2])
with c1:
    refresh = st.button('🔄 Refresh Data', type='primary', use_container_width=True)
if refresh:
    fno_universe.clear(); history.clear(); live.clear(); nifty.clear()
with c2:
    universe = fno_universe()
    symbol = st.selectbox('Search NSE F&O stock', universe, index=None, placeholder='Type symbol, e.g. RELIANCE')
st.caption(f'NSE F&O individual-stock universe: {len(universe)} securities')
with c3:
    if 'updated' in st.session_state:
        st.info('Last Updated\n\n' + st.session_state.updated.strftime('%d-%m-%Y %I:%M:%S %p IST'))

if not symbol:
    st.info('Search and select an NSE F&O stock.')
    st.stop()

with st.spinner(f'Analysing {symbol}...'):
    d = history(symbol)
    n = nifty()
    if d.empty or n.empty:
        st.error('Market data unavailable.')
        st.stop()
    x = make_features(d, n).dropna(subset=['Close', 'E21', 'E50', 'E200'])
    if len(x) < 700:
        st.error('Not enough historical data.')
        st.stop()
    price, chg, pct = live(symbol)
    current_row = live_adjusted_row(x, price)

    results = {}
    for h in HORIZONS:
        res = train_and_predict(x, h, current_row)
        if res is not None:
            results[h] = res
    if not results:
        st.error('Model could not be trained.')
        st.stop()

    # Overall probability is a fixed, transparent weighted combination of the
    # independently calibrated horizon probabilities.
    available_weights = {h: HORIZON_WEIGHTS[h] for h in results}
    total_w = sum(available_weights.values())
    overall = sum(results[h]['prob'] * available_weights[h] for h in results) / total_w

    horizon_dirs = ['Bullish' if results[h]['prob'] >= .60 else 'Bearish' if results[h]['prob'] <= .40 else 'Neutral' for h in results]
    bullish_count = sum(v == 'Bullish' for v in horizon_dirs)
    bearish_count = sum(v == 'Bearish' for v in horizon_dirs)
    if bullish_count >= 2 and overall >= .55:
        direction = 'Bullish'
    elif bearish_count >= 2 and overall <= .45:
        direction = 'Bearish'
    else:
        direction = 'Neutral'

    pos, neg = context(x, price)
    tech_bias, tech_score = technical_bias(x, price)
    horizon_conf = {h: confidence_label(results[h]) for h in results}
    conf_rank = {'Low': 1, 'Moderate': 2, 'High': 3}
    overall_conf = min(horizon_conf.values(), key=lambda z: conf_rank[z]) if horizon_conf else 'Low'
    st.session_state.updated = datetime.now(IST)

r = x.iloc[-1]
live_d = (price - r.E21) / r.E21 * 100 if price is not None else r.D21

st.subheader(symbol)
a, b, c, d4, e4 = st.columns(5)
a.metric('Live Price', f'₹{price:,.2f}' if price is not None else f'₹{r.Close:,.2f}')
b.metric('Change ₹', f'{chg:+,.2f}' if chg is not None else '—')
c.metric('Change %', f'{pct:+.2f}%' if pct is not None else '—')
d4.metric('Direction', direction)
e4.metric('21 EMA Distance', f'{live_d:+.2f}%')

st.subheader('Overall Direction Probability')
ocol1, ocol2, ocol3 = st.columns(3)
ocol1.metric('Overall Upside Probability', f'{overall * 100:.1f}%')
ocol2.metric('Overall Downside Probability', f'{(1 - overall) * 100:.1f}%')
ocol3.metric('Horizon Agreement', f'{max(bullish_count, bearish_count)}/{len(results)}')

st.subheader('Final Assessment')
fa1, fa2, fa3, fa4 = st.columns(4)
fa1.metric('Model Direction', direction)
fa2.metric('Technical Bias', tech_bias)
fa3.metric('Model Confidence', overall_conf)
fa4.metric('Horizon Agreement', f'{max(bullish_count, bearish_count)}/{len(results)}')

st.caption('Model Direction is based on calibrated 5D/10D/20D probabilities. Technical Bias is a separate rule-based summary of the current technical state. Model Confidence describes historical holdout reliability; it is not a guarantee of future returns.')

st.subheader('Directional Probability by Horizon')
rows = []
for h in HORIZONS:
    if h not in results:
        continue
    res = results[h]
    rows.append({
        'Horizon': f'{h} trading days',
        'Upside Probability': f"{res['prob'] * 100:.1f}%",
        'Downside Probability': f"{(1 - res['prob']) * 100:.1f}%",
        'Direction': 'Bullish' if res['prob'] >= .60 else 'Bearish' if res['prob'] <= .40 else 'Neutral',
        'Historical Samples': res['samples'],
        'Validation Accuracy': f"{res['accuracy'] * 100:.1f}%",
        'Brier Score': f"{res['brier']:.3f}",
        'Model Confidence': confidence_label(res),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

q1, q2, q3, q4, q5 = st.columns(5)
q1.metric('RSI', f'{r.RSI:.1f}')
q2.metric('ADX', f'{r.ADX:.1f}')
q3.metric('ATR %', f'{r.ATRp:.2f}%')
q4.metric('Volatility', f'{r.V20:.1f}%')
q5.metric('Volume Ratio', f'{r.VR:.2f}x')

p1, p2 = st.columns(2)
with p1:
    st.markdown('**Positive factors**')
    if pos:
        for z in pos:
            st.write('✅ ' + z)
    else:
        st.write('— None of the selected positive factors are currently active.')
with p2:
    st.markdown('**Risk / negative factors**')
    if neg:
        for z in neg:
            st.write('⚠️ ' + z)
    else:
        st.write('— None of the selected negative factors are currently active.')

st.subheader('EMA Structure')
emas = pd.DataFrame({'EMA': ['21', '50', '100', '200'], 'Value': [r.E21, r.E50, r.E100, r.E200]})
emas['Value'] = emas.Value.map(lambda z: f'₹{z:,.2f}')
st.dataframe(emas, use_container_width=True, hide_index=True)

st.caption('Final version: NSE F&O universe is fetched dynamically with no manual stock list. Training labels exclude observations without a known future outcome. Each horizon is trained and evaluated chronologically, then calibrated from an earlier validation period. The current live price updates close-dependent features for the present prediction, while incomplete intraday High/Low/Volume data are not used to fabricate full-day indicators. Probabilities are estimates, not guarantees, and can be affected by news, gaps and regime changes.')
