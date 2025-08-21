import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import random
import plotly.express as px
import plotly.graph_objects as go

# File paths
TICKERS_FILE = 'tickers.txt'
CACHE_FILE = 'analysis_cache.csv'
PORTFOLIO_FILE = 'portfolio.json'
ERROR_LOG_FILE = 'error_log.txt'

# Load tickers
def load_tickers():
    if not os.path.exists(TICKERS_FILE):
        raise FileNotFoundError(f"{TICKERS_FILE} not found. Please create it with one ticker per line.")
    with open(TICKERS_FILE, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

# Log errors to file
def log_error(message):
    with open(ERROR_LOG_FILE, 'a') as f:
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

# Fetch data with retry mechanism (5 years to ensure enough data for SMA200)
def get_data(ticker, retries=3):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5y', interval='1wk')
            if hist.empty:
                raise ValueError(f"No historical data for {ticker}")
            info = stock.info
            time.sleep(random.uniform(1, 2))
            return hist, info
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            log_error(f"Failed to fetch data for {ticker}: {e}")
            return None, None

# RSI calculation
def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    return 100 - (100 / (1 + rs))

# Technical scoring: SMA200, MACD, RSI, Volume
def technical_score(hist):
    if len(hist) < 100:  # Reduced from 200 to ~2 years of weekly data
        return 5.0
    
    close = hist['Close']
    sma200 = close.rolling(window=200).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    sma_score = 1 if current_price > sma200 else -1
    
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    macd_score = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
    
    rsi_val = rsi(close, 14).iloc[-1]
    rsi_score = 1 if rsi_val > 50 else -1
    
    volume = hist['Volume']
    vol_sma50 = volume.rolling(window=50).mean().iloc[-1]
    vol_score = 1 if volume.iloc[-1] > vol_sma50 else -1
    
    raw_score = (sma_score + macd_score + rsi_score + vol_score) / 4
    normalized_score = (raw_score + 1) * 5
    return normalized_score

# EMA 200 crossover signal
def ema200_signal(hist):
    close = hist['Close']
    ema200 = close.ewm(span=200, adjust=False).mean()
    current_price = close.iloc[-1]
    prev_price = close.iloc[-2]
    current_ema = ema200.iloc[-1]
    prev_ema = ema200.iloc[-2]
    
    if current_price > current_ema and prev_price <= prev_ema:
        return 'Buy'
    elif current_price < current_ema and prev_price >= prev_ema:
        return 'Sell'
    elif current_price > current_ema:
        return 'Bullish'
    elif current_price < current_ema:
        return 'Bearish'
    else:
        return 'Neutral'

# MACD crossover signal
def macd_signal(hist):
    close = hist['Close']
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    current_macd = macd_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    current_signal = signal_line.iloc[-1]
    prev_signal = signal_line.iloc[-2]
    
    if current_macd > current_signal and prev_macd <= prev_signal:
        return 'Buy'
    elif current_macd < current_signal and prev_macd >= prev_signal:
        return 'Sell'
    elif current_macd > current_signal:
        return 'Bullish'
    elif current_macd < current_signal:
        return 'Bearish'
    else:
        return 'Neutral'

# UT Bot Alert Signal (sensitivity a=3, weekly data) - Adjusted to match TradingView
def ut_bot_signal(hist, a=3, c=10):
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    
    tr = pd.DataFrame(index=hist.index)
    tr['tr1'] = high - low
    tr['tr2'] = abs(high - close.shift(1))
    tr['tr3'] = abs(low - close.shift(1))
    tr['tr'] = tr.max(axis=1)
    atr = tr['tr'].rolling(window=c).mean()
    
    nloss = a * atr
    
    ts = pd.Series(np.nan, index=hist.index)
    pos = pd.Series(0, index=hist.index)
    ts[0] = close[0]
    for i in range(1, len(hist)):
        if close[i] > ts[i-1] and close[i-1] > ts[i-1]:
            ts[i] = max(ts[i-1], close[i] - nloss[i])
        elif close[i] < ts[i-1] and close[i-1] < ts[i-1]:
            ts[i] = min(ts[i-1], close[i] + nloss[i])
        elif close[i] > ts[i-1]:
            ts[i] = close[i] - nloss[i]
        else:
            ts[i] = close[i] + nloss[i]
        
        if close]
        
        if close[i-1] < ts[i-1] < ts[i-1] and close[i-1] and close[i][i] > ts[i-1 > ts[i-1]:
            pos[i]]:
            pos[i] = 1
        elif = 1
        elif close[i-1] > close[i-1] > ts[i-1] ts[i-1] and close[i] and close[i] < ts[i-1 < ts[i-1]:
            pos[i] = -]:
            pos[i] = -1
        else1
        else:
            pos[i] = pos[i:
            pos[i] = pos[i-1]
-1    
    above]
    
    above = (close = (close > ts > ts.shift(.shift(1)) &1)) & ( (close.shiftclose.shift(1) <= ts(1) <= ts.shift(1))
    below.shift(1))
    below = (close = (close < ts.shift < ts.shift(1))(1)) & (close.shift & (close.shift(1)(1) >= ts.shift >= ts.shift(1))
(1))
    
    
    buy    buy = (close = (close > ts > ts) &) & above
    above sell
    sell = (close = (close  < ts) &< ts) & below
    
 below
    
    if    if buy.iloc buy.iloc[-1[-1]:
        return 'Buy]:
        return 'Buy'
   '
    elif sell.iloc elif sell.iloc[-1]:
        return '[-1]:
        return 'Sell'
   Sell elif pos'
    elif pos.iloc[-1] == .iloc[-1] == 1:
        return 'Bull1:
        return 'Bullish'
    elif pos.iloc'
    elif pos.iloc[-1] == -1[-1] == -1:
        return 'Bear:
        return 'Bearish'
    elseish'
    else:
        return 'Neutral:
        return 'Neutral'

# Fundamental'

# Fundamental scoring: scoring: Sector comparison
def fundamental Sector comparison
def fundamental_score(info_score(info, sector, sector_med_mediansians):
    sector):
    sector = info = info.get('sector', 'Unknown.get('sector', 'Unknown')
    scores =')
    scores = []
    
    pe []
    
    pe = = info info.get('tr.get('trailingPE', np.nanailingPE', np.nan)
    sector)
    sector_pe = sector_pe = sector_med_medians.get(sector,ians.get(sector, {}).get('tr {}).get('trailingPE', np.nanailingPE', np.nan)
    if np)
    if np.isnan(pe).isnan(pe) or np or np.isnan(sector.isnan(sector_pe_pe):
        scores):
        scores.append(0)
    elif.append(0)
    elif pe < sector pe < sector_pe * 0_pe * 0.8:
        scores.8.append:
        scores.append(2(2)
)
    elif    elif pe < sector pe < sector_pe_pe:
        scores.append(1:
        scores.append(1)
    elif pe < sector)
    elif pe < sector_pe * _pe * 1.12.2:
:
        scores        scores.append(0)
    else.append(0)
    else:
        scores.append(-:
        scores.append(-1)
    
    earnings1)
    
    earnings_growth = info.get('ear_growth = info.get('earningsGrowth', ningsGrowth', 0)0 * 100) * 100
    sector
    sector__eg = sectoreg = sector_medians.get(sector,_medians.get(sector, {}).get {}).('getear('earningsGrowth', ningsGrowth', 0)0) *  * 100
    if100
    if earnings_growth > earnings_growth >  0 and not0 and not np.isnan(pe np.isnan(pe):
        peg =):
        peg = pe / earnings pe / earnings_growth
        sector_growth
        sector_peg = sector_peg = sector_pe / sector_pe / sector_eg if_eg if sector_eg sector_eg > 0 else np > 0 else np.nan
        if np.nan
        if np.isnan(sector_peg.isnan(sector_peg):
            scores):
            scores.append(0.append(0        elif)
        elif peg  peg < sector_peg< sector_peg * * 0.8 0.8:
            scores.append(2:
            scores.append(2)
)
        elif        elif peg < sector peg < sector_peg_peg:
            scores.append(:
            scores.append(1)
1        elif peg)
        elif peg < sector_ < sector_peg * 1peg * 1..2:
            scores.append2:
            scores.append(0)
        else(0)
        else:
            scores.append(-:
            scores.append(-1)
    else1)
    else:
        scores.append(:
        scores.append(0 if earnings0 if earnings_growth > _growth > 00 else - else -1)
    
   1 debt)
    
    debt_equity =_equity = info.get('debtTo info.get('debtToEquity', np.nanEquity', np.nan)
    sector_de = sector)
    sector_de = sector_medians.get(se_medians.get(sector, {}).getctor('debt, {}).get('debtToEquity', np.nanToEquity', np.nan)
    if np.isnan(de)
    if np.isnan(debt_equity)bt_equity) or np.isnan(sector_de or np.isnan(sector_de