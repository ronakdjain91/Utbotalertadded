import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time

# File paths
TICKERS_FILE = 'tickers.txt'  # Input file with one ticker per line
PORTFOLIO_FILE = 'portfolio.json'  # For saving paper trading state (optional, since we use session_state)

# Load tickers
def load_tickers():
    if not os.path.exists(TICKERS_FILE):
        raise FileNotFoundError(f"{TICKERS_FILE} not found. Please create it with one ticker per line.")
    with open(TICKERS_FILE, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

# Fetch data with error handling (weekly data)
def get_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y', interval='1wk')  # 5 years, weekly data
    if hist.empty:
        raise ValueError(f"No historical data for {ticker}")
    info = stock.info
    return hist, info

# RSI calculation
def rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)  # Handle division by zero
    return 100 - (100 / (1 + rs))

# Technical scoring: SMA200, MACD, RSI, Volume
# Score normalized to 0-10
def technical_score(hist):
    if len(hist) < 200:
        return 5.0  # Neutral if insufficient data
    
    close = hist['Close']
    sma200 = close.rolling(window=200).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    sma_score = 1 if current_price > sma200 else -1
    
    # MACD calculation
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    macd_score = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
    
    # RSI score
    rsi_val = rsi(close, 14).iloc[-1]
    rsi_score = 1 if rsi_val > 50 else -1
    
    # Volume score
    volume = hist['Volume']
    vol_sma50 = volume.rolling(window=50).mean().iloc[-1]
    vol_score = 1 if volume.iloc[-1] > vol_sma50 else -1
    
    # Combined score: Average of 4 criteria, shifted to 0-10 scale
    raw_score = (sma_score + macd_score + rsi_score + vol_score) / 4
    normalized_score = (raw_score + 1) * 5  # From 0 to 10
    return normalized_score

# UT Bot Alert Signal (translated from PineScript, sensitivity a=3, on weekly data)
def ut_bot_signal(hist, a=3, c=10):
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    
    # Compute True Range and ATR
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    atr = tr.rolling(window=c).mean()
    
    nloss = a * atr
    
    # Trailing Stop (xATRTrailingStop)
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
    
    # Position (pos)
    pos = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close[i-1] < ts[i-1] and close[i] > ts[i-1]:
            pos[i] = 1
        elif close[i-1] > ts[i-1] and close[i] < ts[i-1]:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]
    
    # Crossovers for buy/sell
    above = (close > ts.shift(1)) & (close.shift(1) <= ts.shift(1))
    below = (close < ts.shift(1)) & (close.shift(1) >= ts.shift(1))
    
    buy = (close > ts) & above
    sell = (close < ts) & below
    
    # Latest signal
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
# Score each stock relative to its sector's median metrics
def fundamental_score(ticker, info, sector_medians):
    sector = info.get('sector', 'Unknown')
    scores = []
    
    # 1. P/E Ratio
    pe = info.get('trailingPE', np.nan)
    sector_pe = sector_medians.get(sector, {}).get('trailingPE', np.nan)
    if np.isnan(pe) or np.isnan(sector_pe):
        scores.append(0)
    elif pe < sector_pe * 0.8:  # Better than sector
        scores.append(2)
    elif pe < sector_pe:
        scores.append(1)
    elif pe < sector_pe * 1.2:
        scores.append(0)
    else:
        scores.append(-1)
    
    # 2. PEG Ratio
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
    
    # 3. Debt to Equity Ratio
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
    
    # 4. Return on Equity (ROE)
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
    
    # 5. Dividend Yield
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
    
    # 6. Revenue Growth
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
    
    # Total raw score (max 12, min -6)
    total_score = sum(scores)
    
    # Normalize to 0-10
    normalized = ((total_score + 6) / 18) * 10
    return max(0, min(10, normalized))

# Compute sector medians
def compute_sector_medians(tickers):
    sector_data = {}
    for ticker in tickers:
        try:
            _, info = get_data(ticker)
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
        except:
            continue
        time.sleep(0.5)  # Small delay to avoid rate limits during median fetch
    
    # Compute medians
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

# Strategy: Combined scoring with sector-adjusted fundamentals
# Weights: Fundamental (50%), Technical (30%), UT Bot (20%)
def get_recommendation(f_score, t_score, ut_signal):
    # Convert UT Bot signal to numeric score
    ut_score = {'Buy': 2, 'Bullish': 1, 'Neutral': 0, 'Bearish': -1, 'Sell': -2}.get(ut_signal, 0)
    # Normalize UT score to 0-10
    ut_normalized = (ut_score + 2) * 2.5  # From -2/+2 to 0/10
    
    # Weighted average
    weighted_score = (0.5 * f_score) + (0.3 * t_score) + (0.2 * ut_normalized)
    
    if weighted_score > 7:
        return 'Buy'
    elif weighted_score < 3:
        return 'Sell'
    else:
        return 'Hold'

# Analyze stocks with batching and progress in Streamlit
def analyze_stocks(tickers):
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(tickers)
    processed = 0
    batch_size = 50  # Batch size to avoid rate limits
    
    # Compute sector medians
    with st.spinner("Computing sector medians..."):
        sector_medians = compute_sector_medians(tickers)
    
    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        for ticker in batch:
            try:
                hist, info = get_data(ticker)
                t_score = technical_score(hist)
                f_score = fundamental_score(ticker, info, sector_medians)
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
                st.warning(f"Error processing {ticker}: {e}")
                continue
            
            processed += 1
            progress = processed / total
            progress_bar.progress(progress)
            status_text.text(f"Processing {processed}/{total}: {ticker}")
        
        if batch_start + batch_size < total:
            status_text.text("Pausing 30 seconds to avoid rate limits...")
            time.sleep(30)  # Delay between batches
    
    status_text.text("Analysis complete!")
    return pd.DataFrame.from_dict(results, orient='index')

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
        except:
            st.warning(f"Error fetching price for {ticker}")
    
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

# Main Streamlit App
st.set_page_config(page_title="Stock Analysis & Paper Trading App (Weekly Data, Sector Comparison)", layout="wide")

st.title("Stock Analysis & Paper Trading App (Weekly Data, Sector Comparison)")

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
    
    if st.button("Refresh Data"):
        with st.spinner("Refreshing data... This may take a while for 750 tickers (batched to avoid rate limits)."):
            df = analyze_stocks(tickers)
            st.session_state.df = df  # Cache in session state
    
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        st.info("Click 'Refresh Data' to analyze stocks.")
        st.stop()
    
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
    
    # Reorder columns to make UT Bot Signal third
    columns = ['Sector', 'Fundamental Score', 'UT Bot Signal', 'Technical Score', 'Recommendation', 'Price', 'TradingView Link']
    filtered_df = filtered_df[columns]
    
    # Display interactive table
    st.subheader("Filtered Recommendations")
    def make_clickable(link):
        if link:
            return f'<a href="{link}" target="_blank">Chart</a>'
        return ''
    
    styled_df = filtered_df.style.format({'TradingView Link': make_clickable})
    st.write(styled_df, unsafe_allow_html=True)
    
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
st.markdown("**Note:** This app uses yfinance for live weekly data. Sectors and fundamentals are fetched from Yahoo Finance. Fundamental scores are sector-adjusted. UT Bot uses sensitivity a=3. Run with `streamlit run app.py`.")
st.markdown("Analysis may take time due to batching (50 tickers, 30s pause) to avoid rate limits.")
