import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title='NSE F&O Direction Analyzer', page_icon='📈', layout='wide')
IST=ZoneInfo('Asia/Kolkata')
@st.cache_data(ttl=3600, show_spinner=False)
def fno_universe():
    """Fetch the NSE F&O individual-security universe dynamically.

    No hardcoded stock list is used. The NSE API is the primary source and the
    official NSE underlyings page is used only as a dynamic fallback.
    """
    index_symbols = {'NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY','NIFTYNXT50'}
    allowed = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&.')

    def clean(values):
        out=[]
        for value in values:
            if not isinstance(value,str):
                continue
            symbol=value.strip().upper()
            if not (2 <= len(symbol) <= 25):
                continue
            if symbol in index_symbols:
                continue
            if all(ch in allowed for ch in symbol):
                out.append(symbol)
        return sorted(set(out))

    # Primary: NSE's live underlying-information API.
    try:
        import requests
        s=requests.Session()
        s.headers.update({
            'User-Agent':'Mozilla/5.0',
            'Accept':'application/json,text/plain,*/*',
            'Referer':'https://www.nseindia.com/'
        })
        s.get('https://www.nseindia.com',timeout=10)
        r=s.get('https://www.nseindia.com/api/underlying-information',timeout=20)
        if r.ok:
            vals=[]
            def walk(x):
                if isinstance(x,list):
                    for v in x: walk(v)
                elif isinstance(x,dict):
                    if isinstance(x.get('symbol'),str):
                        vals.append(x['symbol'])
                    for v in x.values(): walk(v)
            walk(r.json())
            vals=clean(vals)
            if vals:
                return vals
    except Exception:
        pass

    # Dynamic fallback: parse the official NSE page, never a manual list.
    try:
        import requests
        from io import StringIO
        s=requests.Session()
        s.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.nseindia.com/'})
        r=s.get('https://www.nseindia.com/static/products-services/equity-derivatives-list-underlyings-information',timeout=20)
        if r.ok:
            tables=pd.read_html(StringIO(r.text))
            for table in tables:
                cols={str(c).strip().upper():c for c in table.columns}
                symbol_col=next((c for k,c in cols.items() if 'SYMBOL' in k),None)
                if symbol_col is not None:
                    vals=clean(table[symbol_col].tolist())
                    if vals:
                        return vals
    except Exception:
        pass

    return []


@st.cache_data(ttl=900, show_spinner=False)
def history(symbol):
    try:
        d=yf.Ticker(symbol+'.NS').history(period='10y',interval='1d',auto_adjust=False,actions=False)
        if d.empty: return pd.DataFrame()
        d=d[['Open','High','Low','Close','Volume']].dropna(subset=['Close'])
        d.index=pd.to_datetime(d.index).tz_localize(None)
        return d
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def live(symbol):
    try:
        d=yf.Ticker(symbol+'.NS').history(period='1d',interval='5m',auto_adjust=False,prepost=False,actions=False)
        q=d['Close'].dropna()
        if q.empty:return None,None,None
        price=float(q.iloc[-1])
        daily=yf.Ticker(symbol+'.NS').history(period='5d',interval='1d',auto_adjust=False,actions=False)['Close'].dropna()
        prev=float(daily.iloc[-2]) if len(daily)>=2 else float(q.iloc[0])
        return price,price-prev,(price/prev-1)*100
    except Exception:return None,None,None

@st.cache_data(ttl=900, show_spinner=False)
def nifty():
    try:
        d=yf.Ticker('^NSEI').history(period='10y',interval='1d',auto_adjust=False,actions=False)
        d=d[['Open','High','Low','Close','Volume']].dropna(subset=['Close'])
        d.index=pd.to_datetime(d.index).tz_localize(None)
        return d
    except Exception:return pd.DataFrame()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    a=up.ewm(alpha=1/n,adjust=False).mean(); b=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+a/b.replace(0,np.nan))

def features(d, market):
    x=d.copy()
    for k in [10,21,50,100,200]:
        x[f'E{k}']=x.Close.ewm(span=k,adjust=False).mean(); x[f'S{k}']=x[f'E{k}'].pct_change(10)*100
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),(x.Low-x.Close.shift()).abs()],axis=1).max(axis=1)
    x['ATR']=tr.rolling(14).mean(); x['ATRp']=x.ATR/x.Close*100; x['RSI']=rsi(x.Close)
    up=x.High.diff(); dn=-x.Low.diff(); p=up.where((up>dn)&(up>0),0); m=dn.where((dn>up)&(dn>0),0)
    a=tr.ewm(alpha=1/14,adjust=False).mean(); pdi=100*p.ewm(alpha=1/14,adjust=False).mean()/a; mdi=100*m.ewm(alpha=1/14,adjust=False).mean()/a
    x['ADX']=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean(); mac=e12-e26; x['MACD']=mac-mac.ewm(span=9,adjust=False).mean()
    for k in [5,10,21,63,126]: x[f'R{k}']=x.Close.pct_change(k)*100
    x['V20']=x.Close.pct_change().rolling(20).std()*np.sqrt(252)*100; x['V60']=x.Close.pct_change().rolling(60).std()*np.sqrt(252)*100; x['VR']=x.Volume/x.Volume.rolling(20).mean()
    x['D21']=(x.Close-x.E21)/x.E21*100; x['D50']=(x.Close-x.E50)/x.E50*100; x['D200']=(x.Close-x.E200)/x.E200*100
    x['HH20']=x.High.rolling(20).max(); x['LL20']=x.Low.rolling(20).min(); x['BO']=(x.Close>x.HH20.shift()).astype(int); x['BD']=(x.Close<x.LL20.shift()).astype(int)
    market_close = market[['Close']].rename(columns={'Close':'NC'}).reindex(x.index).ffill()
    x=x.join(market_close,how='left')
    x['RS1']=(x['Close'].pct_change(21)-x['NC'].pct_change(21))*100
    x['RS3']=(x['Close'].pct_change(63)-x['NC'].pct_change(63))*100
    x['RS6']=(x['Close'].pct_change(126)-x['NC'].pct_change(126))*100
    market_close_series=x['NC']
    n50=market_close_series.ewm(span=50,adjust=False).mean()
    n200=market_close_series.ewm(span=200,adjust=False).mean()
    x['NRET']=market_close_series.pct_change(21)*100
    x['NV']=market_close_series.pct_change().rolling(20).std()*np.sqrt(252)*100
    x['NT']=(market_close_series>n50).astype(int)
    x['NLT']=(market_close_series>n200).astype(int)
    return x.replace([np.inf,-np.inf],np.nan)

FEATS=['R5','R10','R21','R63','R126','RSI','ADX','MACD','V20','V60','VR','ATRp','D21','D50','D200','S10','S21','S50','S100','S200','BO','BD','RS1','RS3','RS6','NRET','NV','NT','NLT']

def model_prob(x,h):
    y=(x.Close.shift(-h)>x.Close).astype(float); d=x[FEATS].copy(); ok=d.notna().all(axis=1)&y.notna(); d=d.loc[ok]; y=y.loc[ok].astype(int)
    if len(d)<700 or y.nunique()<2:return None,len(d)
    split=int(len(d)*.75); X=d.iloc[:split]; Y=y.iloc[:split]
    m=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=180,learning_rate=.035,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=2,random_state=42))
    m.fit(X,Y); return float(m.predict_proba(x[FEATS].iloc[[-1]])[0,1]),len(d)

def context(x):
    r=x.iloc[-1]; pos=[]; neg=[]
    if r.Close>r.E21>r.E50>r.E200:pos.append('Bullish EMA structure')
    else:neg.append('EMA structure is not fully bullish')
    if r.S21>0 and r.S50>0:pos.append('21/50 EMA rising')
    else:neg.append('21/50 EMA slope is weak')
    if r.RS1>0 and r.RS3>0:pos.append('Outperforming Nifty over 1M and 3M')
    else:neg.append('Relative strength is mixed')
    if r.R21>0 and r.R63>0:pos.append('Positive 1M and 3M momentum')
    else:neg.append('Momentum is mixed')
    if 2<=r.D21<=7:pos.append('Moderate 21 EMA distance')
    elif r.D21>7:neg.append('Price is extended above 21 EMA')
    if r.ADX>=20:pos.append('Trend strength is meaningful')
    else:neg.append('Trend strength is modest')
    if r.VR>=1:pos.append('Volume above 20-day average')
    else:neg.append('Volume below 20-day average')
    if r.BO:pos.append('Recent 20-day breakout')
    if r.BD:neg.append('Recent 20-day breakdown')
    if r.NT:pos.append('Nifty above 50 EMA')
    else:neg.append('Nifty below 50 EMA')
    return pos,neg

st.title('📈 NSE F&O Stock Direction Analyzer')
st.caption('Probability-based directional analysis. Probabilities are estimates, not guarantees.')

c1,c2,c3=st.columns([1.2,2.5,2])
with c1: refresh=st.button('🔄 Refresh Data',type='primary',use_container_width=True)
if refresh:
    fno_universe.clear(); history.clear(); live.clear(); nifty.clear()
with c2: symbol=st.selectbox('Search NSE F&O stock',fno_universe(),index=None,placeholder='Type symbol, e.g. RELIANCE')
st.caption(f"NSE F&O individual-stock universe: {len(fno_universe())} securities")
with c3:
    if 'updated' in st.session_state: st.info('Last Updated\n\n'+st.session_state.updated.strftime('%d-%m-%Y %I:%M:%S %p IST'))

if not symbol: st.info('Search and select an NSE F&O stock.'); st.stop()

with st.spinner(f'Analysing {symbol}...'):
    d=history(symbol); n=nifty()
    if d.empty or n.empty: st.error('Market data unavailable.'); st.stop()
    x=features(d,n).dropna(subset=['Close','E21','E50','E200'])
    if len(x)<700: st.error('Not enough historical data.'); st.stop()
    price,chg,pct=live(symbol)
    probs={}; samples={}
    for h in [5,10,20]:
        p,s=model_prob(x,h)
        if p is not None:probs[h]=p;samples[h]=s
    if not probs:st.error('Model could not be trained.');st.stop()
    pos,neg=context(x); avg=np.mean(list(probs.values()))
    direction='Bullish' if avg>=.60 else 'Bearish' if avg<=.40 else 'Neutral'
    st.session_state.updated=datetime.now(IST)

r=x.iloc[-1]; live_d=(price-r.E21)/r.E21*100 if price else r.D21
st.subheader(symbol)
a,b,c,d4,e4=st.columns(5)
a.metric('Live Price',f'₹{price:,.2f}' if price else f'₹{r.Close:,.2f}')
b.metric('Change ₹',f'{chg:+,.2f}' if chg is not None else '—')
c.metric('Change %',f'{pct:+.2f}%' if pct is not None else '—')
d4.metric('Direction',direction)
e4.metric('21 EMA Distance',f'{live_d:+.2f}%')

st.subheader('Directional Probability')
t=pd.DataFrame([{'Horizon':f'{h} trading days','Upside Probability':f'{probs[h]*100:.1f}%','Downside Probability':f'{(1-probs[h])*100:.1f}%','Direction':'Bullish' if probs[h]>=.60 else 'Bearish' if probs[h]<=.40 else 'Neutral','Historical Samples':samples[h]} for h in [5,10,20] if h in probs])
st.dataframe(t,use_container_width=True,hide_index=True)

q1,q2,q3,q4,q5=st.columns(5)
q1.metric('RSI',f'{r.RSI:.1f}'); q2.metric('ADX',f'{r.ADX:.1f}'); q3.metric('ATR %',f'{r.ATRp:.2f}%'); q4.metric('Volatility',f'{r.V20:.1f}%'); q5.metric('Volume Ratio',f'{r.VR:.2f}x')

p1,p2=st.columns(2)
with p1:
    st.markdown('**Positive factors**')
    for z in pos:st.write('✅ '+z)
with p2:
    st.markdown('**Risk / negative factors**')
    for z in neg:st.write('⚠️ '+z)

st.subheader('EMA Structure')
emas=pd.DataFrame({'EMA':['21','50','100','200'],'Value':[r.E21,r.E50,r.E100,r.E200]})
emas['Value']=emas.Value.map(lambda z:f'₹{z:,.2f}')
st.dataframe(emas,use_container_width=True,hide_index=True)
st.caption('Historical model uses chronological training data. Live price is displayed for recent market changes. Sudden news/events can invalidate technical signals.')
