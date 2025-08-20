import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import random
import plotly.express as px
import threading

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

# Fetch data with retry mechanism (3 years, weekly data)
def get_data(ticker, retries=3):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='3y', interval='1wk')
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
    if len(hist) < 200:
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

# UT Bot Alert Signal (sensitivity a=3, weekly data)
def ut_bot_signal(hist, a=3, c=10):
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    atr = tr.rolling(window=c).mean()
    
    nloss = a * atr
    
    ts = pd.Series(np.nan, index=close.index)
    for i in range(1, len(close)):
        if pd.isna(ts[i-1]):
            ts[i] = close[i] + nloss[i] if close[i] < close[i-1] else close[i] - nloss[i]
        elif close[i] > ts[i-1] and close[i-1] > ts[i-1]:
            ts[i] = max(ts[i-1], close[i] - nloss[i])
        elif close[i] < ts[i-1] and close[i-1] < ts[i-1]:
            ts[i] = min(ts[i-1], close[i] + nloss[i])
        elif close[i] > ts[i-1]:
            ts[i] = close[i] - nloss[i]
        else:
            ts[i] = close[i] + nloss[i]
    
    pos = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close[i-1] < ts[i-1] and close[i] > ts[i-1]:
            pos[i] = 1
        elif close[i-1] > ts[i-1] and close[i] < ts[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]
    
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
    
    # Sequential data fetching
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
    
    # Check for excessive failures
    if len(failed_tickers) / total > 0.1:  # More than 10% failure
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
                rec = get_recommendation(f_score, t_score, ut_signal)
                current_price = hist['Close'].iloc[-1]
                sector = info.get('sector', 'Unknown')
                
                results[ticker] = {
                    'Sector': sector,
                    'Fundamental Score': round(f_score, 2),
                    'Technical Score': round(t_score, 2),
                    'UT Bot Signal': ut_signal,
                    'Recommendation': rec,
                    'Price': round(current_price, 2),
                    'TradingView Link': f"https://www.tradingview.com/chart/?symbol={ticker}" if rec in ['Buy', 'Sell'] else ''
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
        df['Last Updated'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(CACHE_FILE, index_label='Ticker')
        st.session_state.df = df
        status_placeholder.success(f"Background refresh complete! Processed {len(results)}/{total} tickers.")
    else:
        status_placeholder.error("No data processed. Check error_log.txt for details.")
    
    st.session_state.refreshing = False  # Allow new refresh

# Load from cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, index_col='Ticker')
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
        st.dataframe(pd.DataFrame(df_holdings))
    else:
        st.info("No holdings yet.")
    
    st.write(f"Cash: ${cash:.2f}")
    st.write(f"Total Portfolio Value: ${total_value:.2f}")

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
        st.success(f"Bought {shares} shares of {ticker} @ ${price:.2f}")
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
        st.success(f"Sold {shares} shares of {ticker} @ ${price:.2f}")
    except Exception as e:
        st.error(f"Error selling {ticker}: {e}")
        log_error(f"Error selling {ticker}: {e}")

# Main Streamlit App
st.set_page_config(page_title="Stock Analysis & Paper Trading App", layout="wide")

st.title("Stock Analysis & Paper Trading App (3-Year Weekly Data, Sector Comparison)")

# Initialize session state
if 'refreshing' not in st.session_state:
    st.session_state.refreshing = False

# Load tickers
try:
    tickers = load_tickers()
    st.info(f"Loaded {len(tickers)} tickers from {TICKERS_FILE}")
except Exception as e:
    st.error(str(e))
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["Stock Analysis", "Paper Trading"])

with tab1:
    st.header("Stock Analysis")
    
    # Status and progress placeholders
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    
    # Load cache on start
    cached_df = load_cache()
    if cached_df is not None:
        last_updated = cached_df['Last Updated'].iloc[0] if 'Last Updated' in cached_df.columns else "Unknown"
        st.info(f"Showing cached data (last updated: {last_updated}). Click 'Refresh Data' to update in background.")
    
    # Refresh button
    if st.button("Refresh Data", disabled=st.session_state.refreshing):
        st.session_state.refreshing = True
        status_placeholder.info("Background refresh in progress... UI remains responsive.")
        thread = threading.Thread(target=analyze_stocks, args=(tickers, status_placeholder, progress_bar))
        thread.start()
    
    # Load data
    if 'df' in st.session_state:
        df = st.session_state.df
    elif cached_df is not None:
        df = cached_df
    else:
        st.info("No data available. Click 'Refresh Data' to analyze stocks.")
        st.stop()
    
    # Cache expiration check
    if 'Last Updated' in df.columns:
        cache_age = (pd.Timestamp.now() - pd.to_datetime(df['Last Updated'].iloc[0])).total_seconds() / 3600
        if cache_age > 24:
            st.warning("Cached data is over 24 hours old. Please refresh.")
    
    # Filters in sidebar
    st.sidebar.header("Filters")
    unique_sectors = sorted(df['Sector'].unique())
    selected_sectors = st.sidebar.multiselect("Sectors", unique_sectors, default=unique_sectors)
    
    unique_recs = sorted(df['Recommendation'].unique())
    selected_recs = st.sidebar.multiselect("Recommendations", unique_recs, default=unique_recs)
    
    unique_ut = sorted(df['UT Bot Signal'].unique())
    selected_ut = st.sidebar.multiselect("UT Bot Signals", unique_ut, default=unique_ut)
    
    min_f_score, max_f_score = st.sidebar.slider("Fundamental Score Range", 0.0, 10.0, (0.0, 10.0))
    min_t_score, max_t_score = st.sidebar.slider("Technical Score Range", 0.0, 10.0, (0.0, 10.0))
    min_price, max_price = st.sidebar.slider("Price Range", float(df['Price'].min()), float(df['Price'].max()), (float(df['Price'].min()), float(df['Price'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['Sector'].isin(selected_sectors)) &
        (df['Recommendation'].isin(selected_recs)) &
        (df['UT Bot Signal'].isin(selected_ut)) &
        (df['Fundamental Score'].between(min_f_score, max_f_score)) &
        (df['Technical Score'].between(min_t_score, max_t_score)) &
        (df['Price'].between(min_price, max_price))
    ]
    
    # Reorder columns
    columns = ['Sector', 'Fundamental Score', 'UT Bot Signal', 'Technical Score', 'Recommendation', 'Price', 'TradingView Link']
    filtered_df = filtered_df[columns]
    
    # Display table
    st.subheader("Filtered Recommendations")
    def make_clickable(link):
        if link:
            return f'<a href="{link}" target="_blank">Chart</a>'
        return ''
    
    styled_df = filtered_df.style.format({'TradingView Link': make_clickable})
    st.write(styled_df, unsafe_allow_html=True)
    
    # Visualization
    st.subheader("Recommendation Distribution by Sector")
    rec_counts = filtered_df.groupby(['Sector', 'Recommendation']).size().unstack(fill_value=0)
    fig = px.bar(rec_counts, barmode='stack', title="Recommendations by Sector")
    st.plotly_chart(fig)
    
    # Download option
    csv = filtered_df.to_csv()
    st.download_button("Download CSV", csv, "recommendations.csv", "text/csv")

with tab2:
    st.header("Paper Trading")
    initialize_portfolio()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Buy Stock")
        buy_ticker = st.text_input("Ticker to Buy")
        buy_shares = st.number_input("Shares to Buy", min_value=1.0, step=1.0)
        if st.button("Buy"):
            buy_stock(buy_ticker.upper(), buy_shares)
    
    with col2:
        st.subheader("Sell Stock")
        sell_ticker = st.text_input("Ticker to Sell")
        sell_shares = st.number_input("Shares to Sell", min_value=1.0, step=1.0)
        if st.button("Sell"):
            sell_stock(sell_ticker.upper(), sell_shares)
    
    st.subheader("Portfolio")
    view_portfolio()

# Footer
st.markdown("---")
st.markdown("**Note:** Uses yfinance for 3-year weekly data with sequential fetching. Fundamentals are sector-adjusted. UT Bot uses a=3. Run with `streamlit run app.py`. Data cached in 'analysis_cache.csv'.")
st.markdown("**Warning:** Processing 750 tickers may hit rate limits. Check error_log.txt for issues. Reduce ticker count or use an alternative API (e.g., Alpha Vantage) if errors occur.")