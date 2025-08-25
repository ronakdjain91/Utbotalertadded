import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import random
import plotly.graph_objects as go

# File paths
TICKERS_FILE = 'tickers.txt'
CACHE_FILE = 'analysis_cache.csv'
ERROR_LOG_FILE = 'error_log.txt'

# Currency symbol
CURRENCY_SYMBOL = '₹'

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
    if len(hist) < 100:
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

# UT Bot Alert Signal (sensitivity a=3, weekly data)
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
    # Use positional indexing to avoid datetime-index pitfalls
    ts.iloc[0] = close.iloc[0]
    for i in range(1, len(hist)):
        if close.iloc[i] > ts.iloc[i-1] and close.iloc[i-1] > ts.iloc[i-1]:
            ts.iloc[i] = max(ts.iloc[i-1], close.iloc[i] - nloss.iloc[i])
        elif close.iloc[i] < ts.iloc[i-1] and close.iloc[i-1] < ts.iloc[i-1]:
            ts.iloc[i] = min(ts.iloc[i-1], close.iloc[i] + nloss.iloc[i])
        elif close.iloc[i] > ts.iloc[i-1]:
            ts.iloc[i] = close.iloc[i] - nloss.iloc[i]
        else:
            ts.iloc[i] = close.iloc[i] + nloss.iloc[i]
        
        if close.iloc[i-1] < ts.iloc[i-1] and close.iloc[i] > ts.iloc[i-1]:
            pos.iloc[i] = 1
        elif close.iloc[i-1] > ts.iloc[i-1] and close.iloc[i] < ts.iloc[i-1]:
            pos.iloc[i] = -1
        else:
            pos.iloc[i] = pos.iloc[i-1]
    
    above = (close > ts.shift(1)) & (close.shift(1) <= ts.shift(1))
    below = (close < ts.shift(1)) & (close.shift(1) >= ts.shift(1))
    
    buy = (close > ts) & above
    sell = (close < ts) & below
    
    if buy.iloc[-1]:
        return 'Buy'
    elif sell.iloc[-1]:
        return 'Sell'
    elif pos.iloc[-1] == 1:
        return 'Bullish'
    elif pos.iloc[-1] == -1:
        return 'Bearish'
    else:
        return 'Neutral'

# Factor and composite scoring helpers

def compute_max_drawdown(close, lookback=104):
    series = close.tail(lookback)
    rolling_peak = series.cummax()
    drawdown = (series / rolling_peak) - 1.0
    return float(drawdown.min()) if len(drawdown) > 0 else np.nan


def compute_factor_metrics(hist):
    close = hist['Close']
    volume = hist['Volume']
    # Momentum
    mom_12m = close.pct_change(52).iloc[-1] if len(close) >= 53 else np.nan
    mom_6m = close.pct_change(26).iloc[-1] if len(close) >= 27 else np.nan
    mom_3m = close.pct_change(13).iloc[-1] if len(close) >= 14 else np.nan
    # Volatility (annualized from weekly returns over 52 weeks)
    weekly_rets = close.pct_change().dropna()
    vol_ann = weekly_rets.tail(52).std() * np.sqrt(52) if len(weekly_rets) > 0 else np.nan
    # Max drawdown over last ~2 years
    mdd = compute_max_drawdown(close, lookback=104)
    # Liquidity: average weekly dollar volume over last 20 weeks
    adv_20w = (close * volume).rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    # 52W high proximity
    hh_52w = close.rolling(52).max().iloc[-1] if len(close) >= 52 else np.nan
    prox_52w_high = (close.iloc[-1] / hh_52w) if hh_52w and not np.isnan(hh_52w) and hh_52w != 0 else np.nan
    return {
        'Mom 12M': float(mom_12m) if mom_12m is not None else np.nan,
        'Mom 6M': float(mom_6m) if mom_6m is not None else np.nan,
        'Mom 3M': float(mom_3m) if mom_3m is not None else np.nan,
        'Volatility (Ann)': float(vol_ann) if vol_ann is not None else np.nan,
        'Max Drawdown': float(mdd) if mdd is not None else np.nan,
        'ADV (20w)': float(adv_20w) if adv_20w is not None else np.nan,
        '52W High %': float(prox_52w_high) if prox_52w_high is not None else np.nan
    }


def _pct_rank(series, ascending=True):
    try:
        return series.rank(pct=True, ascending=ascending)
    except Exception:
        return pd.Series(np.nan, index=series.index)


def compute_composite_scores(df):
    # Ensure numeric types
    for col in ['P/E Ratio','PEG Ratio','Debt/Equity','ROE','Dividend Yield','Revenue Growth',
                'Mom 12M','Mom 6M','Volatility (Ann)','Max Drawdown','ADV (20w)','52W High %']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Value: low PE, low D/E, high Dividend Yield
    val_components = []
    if 'P/E Ratio' in df.columns:
        val_components.append(1 - _pct_rank(df['P/E Ratio']).fillna(0.5))
    if 'Debt/Equity' in df.columns:
        val_components.append(1 - _pct_rank(df['Debt/Equity']).fillna(0.5))
    if 'Dividend Yield' in df.columns:
        val_components.append(_pct_rank(df['Dividend Yield'], ascending=False).fillna(0.5))
    df['Score Value'] = (pd.concat(val_components, axis=1).mean(axis=1) if val_components else 0.5) * 100
    # Growth: high Revenue Growth, high Earnings Growth, low PEG
    gro_components = []
    if 'Revenue Growth' in df.columns:
        gro_components.append(_pct_rank(df['Revenue Growth'], ascending=False).fillna(0.5))
    if 'PEG Ratio' in df.columns:
        gro_components.append(1 - _pct_rank(df['PEG Ratio']).fillna(0.5))
    if 'Fundamental Data' in df.columns and 'earningsGrowth' in 'Fundamental Data':
        pass
    # We already have earningsGrowth numeric captured separately if available
    # Quality: high ROE, low D/E
    qlt_components = []
    if 'ROE' in df.columns:
        qlt_components.append(_pct_rank(df['ROE'], ascending=False).fillna(0.5))
    if 'Debt/Equity' in df.columns:
        qlt_components.append(1 - _pct_rank(df['Debt/Equity']).fillna(0.5))
    df['Score Quality'] = (pd.concat(qlt_components, axis=1).mean(axis=1) if qlt_components else 0.5) * 100
    # Momentum: 12M, 6M, proximity to 52W high
    mom_components = []
    if 'Mom 12M' in df.columns:
        mom_components.append(_pct_rank(df['Mom 12M'], ascending=False).fillna(0.5))
    if 'Mom 6M' in df.columns:
        mom_components.append(_pct_rank(df['Mom 6M'], ascending=False).fillna(0.5))
    if '52W High %' in df.columns:
        mom_components.append(_pct_rank(df['52W High %'], ascending=False).fillna(0.5))
    df['Score Momentum'] = (pd.concat(mom_components, axis=1).mean(axis=1) if mom_components else 0.5) * 100
    # Low Volatility: low vol, low max drawdown
    lv_components = []
    if 'Volatility (Ann)' in df.columns:
        lv_components.append(1 - _pct_rank(df['Volatility (Ann)']).fillna(0.5))
    if 'Max Drawdown' in df.columns:
        lv_components.append(1 - _pct_rank(df['Max Drawdown']).fillna(0.5))
    df['Score LowVol'] = (pd.concat(lv_components, axis=1).mean(axis=1) if lv_components else 0.5) * 100
    # Liquidity: high ADV
    liq_components = []
    if 'ADV (20w)' in df.columns:
        liq_components.append(_pct_rank(df['ADV (20w)'], ascending=False).fillna(0.5))
    df['Score Liquidity'] = (pd.concat(liq_components, axis=1).mean(axis=1) if liq_components else 0.5) * 100
    # Composite: blend Value, Growth, Quality, Momentum, LowVol with weights
    weights = {
        'Score Value': 0.2,
        'Score Growth': 0.2,
        'Score Quality': 0.2,
        'Score Momentum': 0.25,
        'Score LowVol': 0.1,
        'Score Liquidity': 0.05
    }
    # Ensure Score Growth exists
    if 'Score Growth' not in df.columns:
        # Approximate growth using revenue growth if needed
        approx = _pct_rank(df.get('Revenue Growth', pd.Series(np.nan, index=df.index)), ascending=False).fillna(0.5) * 100
        df['Score Growth'] = approx
    composite = 0
    for k, w in weights.items():
        if k in df.columns:
            composite = composite + (df[k].fillna(50) * w)
    df['Composite Score'] = composite
    return df


def get_recommendation_v2(row):
    comp = row.get('Composite Score', np.nan)
    ut = row.get('UT Bot Signal', 'Neutral')
    ema = row.get('EMA 200 Signal', 'Neutral')
    price = row.get('Price', np.nan)
    sma200 = row.get('SMA200', np.nan)
    buy_like = (ut in ['Buy','Bullish']) and (ema in ['Buy','Bullish']) and (not np.isnan(price) and not np.isnan(sma200) and price >= sma200)
    sell_like = (ut == 'Sell') or (ema == 'Sell') or (not np.isnan(price) and not np.isnan(sma200) and price < sma200)
    if not np.isnan(comp):
        if comp >= 70 and buy_like:
            return 'Buy'
        elif comp <= 30 and sell_like:
            return 'Sell'
        else:
            return 'Hold'
    return row.get('Recommendation', 'Hold')

# Fundamental scoring: Sector comparison
def fundamental_score(info, sector_medians):
    sector = info.get('sector', 'Unknown')
    scores = []
    
    pe = info.get('trailingPE', np.nan)
    sector_pe = sector_medians.get(sector, {}).get('trailingPE', np.nan)
    if np.isnan(pe) or np.isnan(sector_pe):
        scores.append(0)
    elif pe < sector_pe * 0.8:
        scores.append(2)
    elif pe < sector_pe:
        scores.append(1)
    elif pe < sector_pe * 1.2:
        scores.append(0)
    else:
        scores.append(-1)
    
    earnings_growth = info.get('earningsGrowth', 0) * 100
    sector_eg = sector_medians.get(sector, {}).get('earningsGrowth', 0) * 100
    if earnings_growth > 0 and not np.isnan(pe):
        peg = pe / earnings_growth
        sector_peg = sector_pe / sector_eg if sector_eg > 0 else np.nan
        if np.isnan(sector_peg):
            scores.append(0)
        elif peg < sector_peg * 0.8:
            scores.append(2)
        elif peg < sector_peg:
            scores.append(1)
        elif peg < sector_peg * 1.2:
            scores.append(0)
        else:
            scores.append(-1)
    else:
        scores.append(0 if earnings_growth > 0 else -1)
    
    debt_equity = info.get('debtToEquity', np.nan)
    sector_de = sector_medians.get(sector, {}).get('debtToEquity', np.nan)
    if np.isnan(debt_equity) or np.isnan(sector_de):
        scores.append(0)
    elif debt_equity < sector_de * 0.8:
        scores.append(2)
    elif debt_equity < sector_de:
        scores.append(1)
    elif debt_equity < sector_de * 1.2:
        scores.append(0)
    else:
        scores.append(-1)
    
    roe = info.get('returnOnEquity', 0)
    sector_roe = sector_medians.get(sector, {}).get('returnOnEquity', 0)
    if roe > sector_roe * 1.2:
        scores.append(2)
    elif roe > sector_roe:
        scores.append(1)
    elif roe > sector_roe * 0.8:
        scores.append(0)
    else:
        scores.append(-1)
    
    dividend_yield = info.get('dividendYield', 0)
    sector_dy = sector_medians.get(sector, {}).get('dividendYield', 0)
    if dividend_yield > sector_dy * 1.2:
        scores.append(2)
    elif dividend_yield > sector_dy:
        scores.append(1)
    elif dividend_yield > sector_dy * 0.8:
        scores.append(0)
    else:
        scores.append(-1)
    
    revenue_growth = info.get('revenueGrowth', 0)
    sector_rg = sector_medians.get(sector, {}).get('revenueGrowth', 0)
    if revenue_growth > sector_rg * 1.2:
        scores.append(2)
    elif revenue_growth > sector_rg:
        scores.append(1)
    elif revenue_growth > sector_rg * 0.8:
        scores.append(0)
    else:
        scores.append(-1)
    
    total_score = sum(scores)
    normalized = ((total_score + 6) / 18) * 10
    return max(0, min(10, normalized))

# Compute sector medians
def compute_sector_medians(data_dict):
    sector_data = {}
    for ticker, data in data_dict.items():
        info = data['info']
        sector = info.get('sector', 'Unknown')
        if sector not in sector_data:
            sector_data[sector] = {
                'trailingPE': [],
                'earningsGrowth': [],
                'debtToEquity': [],
                'returnOnEquity': [],
                'dividendYield': [],
                'revenueGrowth': []
            }
        sector_data[sector]['trailingPE'].append(info.get('trailingPE', np.nan))
        sector_data[sector]['earningsGrowth'].append(info.get('earningsGrowth', np.nan))
        sector_data[sector]['debtToEquity'].append(info.get('debtToEquity', np.nan))
        sector_data[sector]['returnOnEquity'].append(info.get('returnOnEquity', np.nan))
        sector_data[sector]['dividendYield'].append(info.get('dividendYield', np.nan))
        sector_data[sector]['revenueGrowth'].append(info.get('revenueGrowth', np.nan))
    
    sector_medians = {}
    for sector, metrics in sector_data.items():
        sector_medians[sector] = {
            'trailingPE': np.nanmedian(metrics['trailingPE']),
            'earningsGrowth': np.nanmedian(metrics['earningsGrowth']),
            'debtToEquity': np.nanmedian(metrics['debtToEquity']),
            'returnOnEquity': np.nanmedian(metrics['returnOnEquity']),
            'dividendYield': np.nanmedian(metrics['dividendYield']),
            'revenueGrowth': np.nanmedian(metrics['revenueGrowth'])
        }
    return sector_medians

# Strategy: Weighted scoring
def get_recommendation(f_score, t_score, ut_signal):
    ut_score = {'Buy': 2, 'Bullish': 1, 'Neutral': 0, 'Bearish': -1, 'Sell': -2}.get(ut_signal, 0)
    ut_normalized = (ut_score + 2) * 2.5
    weighted_score = (0.5 * f_score) + (0.3 * t_score) + (0.2 * ut_normalized)
    if weighted_score > 7:
        return 'Buy'
    elif weighted_score < 3:
        return 'Sell'
    else:
        return 'Hold'

# Analyze stocks with progress bar and stock name
def analyze_stocks(tickers, status_placeholder, progress_bar):
    data_dict = {}
    total = len(tickers)
    failed_tickers = []
    
    status_placeholder.text("Fetching data sequentially...")
    progress_bar.progress(0.0)
    
    for i, ticker in enumerate(tickers):
        try:
            hist, info = get_data(ticker)
            if hist is not None and info is not None:
                data_dict[ticker] = {'hist': hist, 'info': info}
            else:
                failed_tickers.append(ticker)
                status_placeholder.warning(f"Error fetching data for {ticker}")
        except Exception as e:
            status_placeholder.warning(f"Error fetching data for {ticker}: {e}")
            log_error(f"Error fetching data for {ticker}: {e}")
            failed_tickers.append(ticker)
        
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_placeholder.text(f"Fetching data {i + 1}/{total}: {ticker}")
    
    if len(failed_tickers) / total > 0.1:
        status_placeholder.warning("High failure rate detected. Possible rate limit. Check error_log.txt and consider reducing tickers.")
    
    results = {}
    if data_dict:
        status_placeholder.text("Computing sector medians...")
        progress_bar.progress(0.0)
        sector_medians = compute_sector_medians(data_dict)
        
        processed = 0
        for ticker, data in data_dict.items():
            try:
                hist = data['hist']
                info = data['info']
                t_score = technical_score(hist)
                f_score = fundamental_score(info, sector_medians)
                ut_signal = ut_bot_signal(hist, a=3)
                ema_signal = ema200_signal(hist)
                macd_sig = macd_signal(hist)
                rec = get_recommendation(f_score, t_score, ut_signal)
                current_price = hist['Close'].iloc[-1]
                sector = info.get('sector', 'Unknown')
                
                pe = info.get('trailingPE', np.nan)
                earnings_growth = info.get('earningsGrowth', np.nan)
                peg = pe / (earnings_growth * 100) if earnings_growth and earnings_growth > 0 else np.nan
                debt_equity = info.get('debtToEquity', np.nan)
                roe = info.get('returnOnEquity', np.nan)
                dividend_yield = info.get('dividendYield', np.nan)
                revenue_growth = info.get('revenueGrowth', np.nan)
                
                rsi_val = rsi(hist['Close'], 14).iloc[-1]
                macd_val = (hist['Close'].ewm(span=12, adjust=False).mean() - hist['Close'].ewm(span=26, adjust=False).mean()).iloc[-1]
                sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                vol_ratio = hist['Volume'].iloc[-1] / hist['Volume'].rolling(window=50).mean().iloc[-1]
                # New factor metrics
                factors = compute_factor_metrics(hist)
                
                # QTD return
                qtd_ret = compute_qtd_return(hist)
                
                fundamental_data = f"P/E: {round(pe, 2) if not np.isnan(pe) else 'N/A'}, PEG: {round(peg, 2) if not np.isnan(peg) else 'N/A'}, Debt/Equity: {round(debt_equity, 2) if not np.isnan(debt_equity) else 'N/A'}, ROE: {round(roe, 2) if not np.isnan(roe) else 'N/A'}, Dividend Yield: {round(dividend_yield, 2) if not np.isnan(dividend_yield) else 'N/A'}, Revenue Growth: {round(revenue_growth, 2) if not np.isnan(revenue_growth) else 'N/A'}"
                technical_data = f"RSI: {round(rsi_val, 0)}, MACD: {round(macd_val, 2)}, SMA200: {round(sma200, 2)}, Volume vs SMA50: {round(vol_ratio, 0)}"
                
                results[ticker] = {
                    'Sector': sector,
                    'Fundamental Score': round(f_score, 2),
                    'Fundamental Data': fundamental_data,
                    'P/E Ratio': round(pe, 2) if not np.isnan(pe) else np.nan,
                    'PEG Ratio': round(peg, 2) if not np.isnan(peg) else np.nan,
                    'Debt/Equity': round(debt_equity, 2) if not np.isnan(debt_equity) else np.nan,
                    'ROE': round(roe, 2) if not np.isnan(roe) else np.nan,
                    'Dividend Yield': round(dividend_yield, 2) if not np.isnan(dividend_yield) else np.nan,
                    'Revenue Growth': round(revenue_growth, 2) if not np.isnan(revenue_growth) else np.nan,
                    'Technical Score': round(t_score, 2),
                    'Technical Data': technical_data,
                    'RSI': round(rsi_val, 0),
                    'MACD': round(macd_val, 2),
                    'SMA200': round(sma200, 2),
                    'Volume vs SMA50': round(vol_ratio, 0),
                    'UT Bot Signal': ut_signal,
                    'EMA 200 Signal': ema_signal,
                    'MACD Signal': macd_sig,
                    'Recommendation': rec,
                    'Price': round(current_price, 2),
                    'TradingView Link': f"https://www.tradingview.com/chart/?symbol={ticker}",
                    # New factors
                    'Mom 12M': factors['Mom 12M'],
                    'Mom 6M': factors['Mom 6M'],
                    'Mom 3M': factors['Mom 3M'],
                    'Volatility (Ann)': factors['Volatility (Ann)'],
                    'Max Drawdown': factors['Max Drawdown'],
                    'ADV (20w)': factors['ADV (20w)'],
                    '52W High %': factors['52W High %'],
                    'QTD Return': qtd_ret
                }
            except Exception as e:
                status_placeholder.warning(f"Error computing scores for {ticker}: {e}")
                log_error(f"Error computing scores for {ticker}: {e}")
                continue
            
            processed += 1
            progress = processed / len(data_dict)
            progress_bar.progress(progress)
            status_placeholder.text(f"Computing scores {processed}/{len(data_dict)}: {ticker}")
    
    if results:
        df = pd.DataFrame.from_dict(results, orient='index')
        # Compute composite and strategy scores and refresh recommendations
        df = compute_composite_scores(df)
        df['Recommendation'] = df.apply(get_recommendation_v2, axis=1)
        # Strategy flags
        flags_df = df.apply(qualify_strategies, axis=1, result_type='expand')
        if isinstance(flags_df, pd.DataFrame):
            for col in flags_df.columns:
                df[col] = flags_df[col]
        df['Last Updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(CACHE_FILE, index_label='Ticker')
        return df
    else:
        status_placeholder.error("No data processed. Check error_log.txt for details.")
        return None

# Load from cache with column validation
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE, index_col='Ticker')
            expected_columns = [
                'Sector', 'Fundamental Score', 'Fundamental Data', 'P/E Ratio', 'PEG Ratio', 'Debt/Equity', 'ROE',
                'Dividend Yield', 'Revenue Growth', 'Technical Score', 'Technical Data', 'RSI', 'MACD', 'SMA200',
                'Volume vs SMA50', 'UT Bot Signal', 'EMA 200 Signal', 'MACD Signal', 'Recommendation', 'Price',
                'TradingView Link', 'Mom 12M', 'Mom 6M', 'Mom 3M', 'Volatility (Ann)', 'Max Drawdown', 'ADV (20w)', '52W High %',
                'Score Value', 'Score Growth', 'Score Quality', 'Score Momentum', 'Score LowVol', 'Score Liquidity', 'Composite Score', 'QTD Return',
                'Qualifies Value', 'Qualifies Growth', 'Qualifies Quality', 'Qualifies Momentum', 'Qualifies LowVol', 'Qualifies Liquidity', 'Qualifies All-Rounder',
                'Last Updated'
            ]
            for col in expected_columns:
                if col not in df.columns:
                    numeric_cols = ['Fundamental Score', 'P/E Ratio', 'PEG Ratio', 'Debt/Equity', 'ROE', 'Dividend Yield', 'Revenue Growth', 'Technical Score', 'RSI', 'MACD', 'SMA200', 'Volume vs SMA50', 'Price',
                                    'Mom 12M', 'Mom 6M', 'Mom 3M', 'Volatility (Ann)', 'Max Drawdown', 'ADV (20w)', '52W High %',
                                    'Score Value', 'Score Growth', 'Score Quality', 'Score Momentum', 'Score LowVol', 'Score Liquidity', 'Composite Score']
                    df[col] = np.nan if col in numeric_cols else ''
                    log_error(f"Added missing column {col} to cache")
            return df
        except Exception as e:
            log_error(f"Error loading cache: {e}")
            return None
    return None

# Paper trading functions
def initialize_portfolio():
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'cash': 100000.0,
            'holdings': {}
        }

def view_portfolio():
    portfolio = st.session_state.portfolio
    cash = portfolio['cash']
    holdings = portfolio['holdings']
    total_value = cash
    df_holdings = []
    
    for ticker, data in holdings.items():
        try:
            current_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
            value = data['shares'] * current_price
            pnl = value - (data['shares'] * data['avg_buy_price'])
            df_holdings.append({
                'Ticker': ticker,
                'Shares': data['shares'],
                'Avg Buy Price': round(data['avg_buy_price'], 2),
                'Current Price': round(current_price, 2),
                'Value': round(value, 2),
                'P&L': round(pnl, 2)
            })
            total_value += value
        except Exception as e:
            st.warning(f"Error fetching price for {ticker}: {e}")
            log_error(f"Error fetching portfolio price for {ticker}: {e}")
    
    if df_holdings:
        df_portfolio = pd.DataFrame(df_holdings)
        st.subheader("Portfolio Holdings")
        for i, row in df_portfolio.iterrows():
            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
            cols[0].write(row['Ticker'])
            cols[1].write(row['Shares'])
            cols[2].write(f"{CURRENCY_SYMBOL}{row['Avg Buy Price']:.2f}")
            cols[3].write(f"{CURRENCY_SYMBOL}{row['Current Price']:.2f}")
            cols[4].write(f"{CURRENCY_SYMBOL}{row['Value']:.2f}")
            cols[5].write(f"{CURRENCY_SYMBOL}{row['P&L']:.2f}")
            if cols[6].button("Sell", key=f"sell_{row['Ticker']}_{i}"):
                sell_shares = st.number_input(f"Shares to Sell ({row['Ticker']})", min_value=1.0, max_value=row['Shares'], step=1.0, key=f"sell_shares_{row['Ticker']}_{i}")
                if st.button(f"Confirm Sell {row['Ticker']}", key=f"confirm_sell_{row['Ticker']}_{i}"):
                    sell_stock(row['Ticker'], sell_shares)
                    st.experimental_rerun()
    
    st.metric("Cash", f"{CURRENCY_SYMBOL}{cash:.2f}")
    st.metric("Total Portfolio Value", f"{CURRENCY_SYMBOL}{total_value:.2f}")

def buy_stock(ticker, shares):
    portfolio = st.session_state.portfolio
    try:
        price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        cost = shares * price
        if cost > portfolio['cash']:
            st.error("Insufficient cash")
            return
        portfolio['cash'] -= cost
        if ticker in portfolio['holdings']:
            old_shares = portfolio['holdings'][ticker]['shares']
            old_cost = old_shares * portfolio['holdings'][ticker]['avg_buy_price']
            new_shares = old_shares + shares
            new_avg = (old_cost + cost) / new_shares
            portfolio['holdings'][ticker] = {'shares': new_shares, 'avg_buy_price': new_avg}
        else:
            portfolio['holdings'][ticker] = {'shares': shares, 'avg_buy_price': price}
        st.success(f"Bought {shares} shares of {ticker} @ {CURRENCY_SYMBOL}{price:.2f}")
    except Exception as e:
        st.error(f"Error buying {ticker}: {e}")
        log_error(f"Error buying {ticker}: {e}")

def sell_stock(ticker, shares):
    portfolio = st.session_state.portfolio
    if ticker not in portfolio['holdings'] or shares > portfolio['holdings'][ticker]['shares']:
        st.error("Insufficient shares or not held")
        return
    try:
        price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        revenue = shares * price
        portfolio['cash'] += revenue
        portfolio['holdings'][ticker]['shares'] -= shares
        if portfolio['holdings'][ticker]['shares'] <= 0:
            del portfolio['holdings'][ticker]
        st.success(f"Sold {shares} shares of {ticker} @ {CURRENCY_SYMBOL}{price:.2f}")
    except Exception as e:
        st.error(f"Error selling {ticker}: {e}")
        log_error(f"Error selling {ticker}: {e}")

def compute_qtd_return(hist):
    try:
        last_date = hist.index[-1]
        quarter_start_month = ((last_date.month - 1)//3)*3 + 1
        quarter_start = pd.Timestamp(year=last_date.year, month=quarter_start_month, day=1)
        # find first trading point on/after quarter_start
        idx = hist.index.get_indexer([quarter_start], method='bfill')[0]
        start_price = hist['Close'].iloc[idx]
        end_price = hist['Close'].iloc[-1]
        return float((end_price / start_price) - 1.0) if start_price and start_price != 0 else np.nan
    except Exception:
        return np.nan


# Threshold settings helpers

def get_strategy_thresholds():
    defaults = {
        'thr_value': 60,
        'thr_growth': 60,
        'thr_quality': 60,
        'thr_momentum': 60,
        'thr_lowvol': 60,
        'thr_liquidity': 60,
        'thr_allrounder': 70,
        'require_bullish': True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    return {k: st.session_state[k] for k in defaults.keys()}


def qualify_strategies(row):
    flags = {}
    thr = get_strategy_thresholds()
    price = row.get('Price', np.nan)
    sma200 = row.get('SMA200', np.nan)
    ut = row.get('UT Bot Signal', 'Neutral')
    ema = row.get('EMA 200 Signal', 'Neutral')
    bullish = True
    if thr.get('require_bullish', True):
        bullish = (ut in ['Buy','Bullish']) and (ema in ['Buy','Bullish']) and (not np.isnan(price) and not np.isnan(sma200) and price >= sma200)
    # Thresholds
    flags['Qualifies Value'] = (row.get('Score Value', 0) >= thr.get('thr_value', 60)) and bullish
    flags['Qualifies Growth'] = (row.get('Score Growth', 0) >= thr.get('thr_growth', 60)) and bullish
    flags['Qualifies Quality'] = (row.get('Score Quality', 0) >= thr.get('thr_quality', 60)) and bullish
    flags['Qualifies Momentum'] = (row.get('Score Momentum', 0) >= thr.get('thr_momentum', 60)) and bullish
    flags['Qualifies LowVol'] = (row.get('Score LowVol', 0) >= thr.get('thr_lowvol', 60)) and bullish
    flags['Qualifies Liquidity'] = (row.get('Score Liquidity', 0) >= thr.get('thr_liquidity', 60)) and bullish
    flags['Qualifies All-Rounder'] = (row.get('Composite Score', 0) >= thr.get('thr_allrounder', 70)) and bullish
    return flags

# Main Streamlit App
st.set_page_config(page_title="Stock Analysis & Paper Trading App", layout="wide")

# Custom CSS for improved styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0b1221;
        color: #e5e7eb;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        color: white;
        border-radius: 8px;
        border: 0;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .stTextInput>div>input {
        border-radius: 8px;
    }
    .stMetric {
        background-color: #111827;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content {
        background-color: #111827;
        border-radius: 8px;
        padding: 10px;
    }
    a { color: #93c5fd; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Stock Analysis & Paper Trading App")
st.markdown("Analyze stocks with 5-year weekly data and sector-adjusted fundamentals. Trade virtually with ₹100,000 starting capital.")

# Initialize portfolio
initialize_portfolio()

# Load tickers
try:
    tickers = load_tickers()
    st.info(f"Loaded {len(tickers)} tickers from {TICKERS_FILE}")
except Exception as e:
    st.error(str(e))
    st.stop()

# Status and progress placeholders
with st.container():
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)

# Load cache on start
cached_df = load_cache()
if cached_df is not None:
    last_updated = cached_df['Last Updated'].iloc[0] if 'Last Updated' in cached_df.columns else "Unknown"
    st.info(f"Showing cached data (last updated: {last_updated}). Click 'Refresh Data' to update.")

# Ensure df is available
if 'df' in st.session_state:
    df = st.session_state.df
elif cached_df is not None:
    df = cached_df
else:
    st.info("No data available. Click 'Refresh Data' to analyze stocks.")
    st.stop()

# Sidebar no longer uses global filters; provide quick actions instead
st.sidebar.header("⚙️ Actions")
st.sidebar.write("Use tabs below to explore. Refresh data when needed.")
if st.sidebar.button("🔄 Refresh Data"):
    status_placeholder.info("Processing data... UI will be unresponsive until complete.")
    df_ref = analyze_stocks(tickers, status_placeholder, progress_bar)
    if df_ref is not None:
        st.session_state.df = df_ref
        st.experimental_rerun()
    else:
        st.error("Analysis failed. Check error_log.txt for details.")

# Tabs
overview_tab, matrix_tab, backtest_tab, full_tab, portfolio_tab, settings_tab = st.tabs(["Overview","Strategy Matrix","Backtest","Full Data","Portfolio","Settings"])

with overview_tab:
    st.subheader("Strategy-wise Recommendations")
    # Recompute strategy flags with current thresholds
    data = df.copy()
    flags_df = data.apply(qualify_strategies, axis=1, result_type='expand')
    if isinstance(flags_df, pd.DataFrame):
        for col in flags_df.columns:
            data[col] = flags_df[col]
    strategy_cols = [c for c in data.columns if c.startswith('Qualifies ')]
    counts = []
    for c in strategy_cols:
        counts.append({"Strategy": c.replace('Qualifies ', ''), "Buys": int(data[c].fillna(False).sum())})
    if counts:
        chart_df = pd.DataFrame(counts)
        fig = go.Figure(data=[go.Bar(x=chart_df['Strategy'], y=chart_df['Buys'], marker_color="#22c55e")])
        fig.update_layout(title="Buy Counts by Strategy", template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True)
    # Sector QTD performance
    st.markdown("**Sector QTD Performance**")
    if 'QTD Return' in data.columns and 'Sector' in data.columns:
        sector_qtd = data.groupby('Sector')['QTD Return'].median().sort_values(ascending=False)
        booming_sector = sector_qtd.index[0] if len(sector_qtd) else 'N/A'
        st.metric("Booming Sector (QTD median)", booming_sector)
        st.dataframe(sector_qtd.reset_index().rename(columns={'QTD Return':'QTD Median'}))
    # Top 10 All-Rounder picks
    if 'Composite Score' in data.columns:
        top_all = data.sort_values('Composite Score', ascending=False).head(10)
        st.markdown("**Top 10 All-Rounder Picks**")
        st.dataframe(top_all[['Sector','Price','Composite Score','Score Quality','Score Momentum','Score Value','Score Growth']].style.format({'Price': f'{CURRENCY_SYMBOL}{{:.2f}}'}))

with matrix_tab:
    st.subheader("Strategy Matrix (which strategies each stock qualifies for)")
    show_only_bullish = st.checkbox("Show only bullish (UT/EMA + above SMA200)", value=False, key="matrix_bullish")
    data = df.copy()
    flags_df = data.apply(qualify_strategies, axis=1, result_type='expand')
    if isinstance(flags_df, pd.DataFrame):
        for col in flags_df.columns:
            data[col] = flags_df[col]
    if show_only_bullish:
        cond = (data['UT Bot Signal'].isin(['Buy','Bullish'])) & (data['EMA 200 Signal'].isin(['Buy','Bullish'])) & (data['Price'] >= data['SMA200'])
        data = data[cond]
    matrix_cols = ['Qualifies All-Rounder','Qualifies Value','Qualifies Growth','Qualifies Quality','Qualifies Momentum','Qualifies LowVol','Qualifies Liquidity']
    present_cols = [c for c in matrix_cols if c in data.columns]
    table = data[present_cols].fillna(False).astype(bool)
    table = table.replace({True: '✓', False: '✗'})
    st.dataframe(table)

with backtest_tab:
    st.subheader("Cohort Performance (Strategy vs Universe)")
    data = df.copy()
    flags_df = data.apply(qualify_strategies, axis=1, result_type='expand')
    if isinstance(flags_df, pd.DataFrame):
        for col in flags_df.columns:
            data[col] = flags_df[col]
    strategy_cols = ['Qualifies All-Rounder','Qualifies Value','Qualifies Growth','Qualifies Quality','Qualifies Momentum','Qualifies LowVol','Qualifies Liquidity']
    periods = [('Mom 3M','3M'), ('Mom 6M','6M'), ('Mom 12M','12M')]
    for metric, label in periods:
        if metric not in data.columns:
            continue
        universe_median = float(pd.to_numeric(data[metric], errors='coerce').median()) if len(data) else np.nan
        rows = []
        for col in strategy_cols:
            if col in data.columns:
                subset = data[data[col] == True]
                med = float(pd.to_numeric(subset[metric], errors='coerce').median()) if len(subset) else np.nan
                rows.append({'Strategy': col.replace('Qualifies ', ''), 'Median Return': med})
        if rows:
            chart_df = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=chart_df['Strategy'], y=chart_df['Median Return'], name=f"Strategy {label}", marker_color="#38bdf8"))
            if not np.isnan(universe_median):
                fig.add_trace(go.Scatter(x=chart_df['Strategy'], y=[universe_median]*len(chart_df), name="Universe Median", mode='lines', line=dict(color="#f43f5e", dash='dash')))
            fig.update_layout(title=f"Median {label} Return", template='plotly_white', height=360)
            st.plotly_chart(fig, use_container_width=True)

with full_tab:
    st.subheader("Complete Data")
    st.dataframe(df)

with portfolio_tab:
    view_portfolio()

with settings_tab:
    st.subheader("Strategy Thresholds & Signals")
    thr = get_strategy_thresholds()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state['thr_value'] = st.slider("Value threshold", 0, 100, thr['thr_value'])
        st.session_state['thr_quality'] = st.slider("Quality threshold", 0, 100, thr['thr_quality'])
    with c2:
        st.session_state['thr_growth'] = st.slider("Growth threshold", 0, 100, thr['thr_growth'])
        st.session_state['thr_momentum'] = st.slider("Momentum threshold", 0, 100, thr['thr_momentum'])
    with c3:
        st.session_state['thr_lowvol'] = st.slider("LowVol threshold", 0, 100, thr['thr_lowvol'])
        st.session_state['thr_liquidity'] = st.slider("Liquidity threshold", 0, 100, thr['thr_liquidity'])
    st.session_state['thr_allrounder'] = st.slider("All-Rounder Composite threshold", 0, 100, thr['thr_allrounder'])
    st.session_state['require_bullish'] = st.checkbox("Require bullish confirmation (UT/EMA + price ≥ SMA200)", value=thr['require_bullish'])
    st.info("Thresholds apply immediately to the Strategy Matrix and Overview counts.")

# Download option
st.download_button(
    label="📥 Download CSV",
    data=df.to_csv(),
    file_name="recommendations.csv",
    mime="text/csv",
    help="Download the complete data as a CSV file"
)

# Footer
st.markdown("---")
st.markdown(f"""
    **Note:** Uses yfinance for 5-year weekly data. Fundamentals are sector-adjusted. UT Bot uses a=3. Run with `streamlit run App.py`. Data cached in 'analysis_cache.csv'.
    **Warning:** Processing 752 tickers may hit rate limits. Check error_log.txt for issues. Reduce ticker count or use an alternative API (e.g., Alpha Vantage) if errors occur.
    **Developed by:** Your Name | Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
