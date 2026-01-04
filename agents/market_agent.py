# agents/market_agent.py
import requests  # For API calls
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
import time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential  # Update to exponential backoff for better rate limit handling
from dotenv import load_dotenv
import os
import pandas as pd  # For handling historical data as DataFrame

load_dotenv()  # Load environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env file")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Use a global dict for caching with TTL
cache = {}  # {ticker: {"data": ..., "timestamp": datetime}}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))  # Up to 5 retries with exponential backoff (4s, 8s, 16s, etc.)
def get_stock_data(ticker):
    now = datetime.now()
    if ticker in cache and now - cache[ticker]["timestamp"] < timedelta(minutes=15):  # Increase TTL to 15 min to reduce API calls
        return cache[ticker]

    try:
        # Get overview (fundamentals)
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url).json()
        if "Note" in overview_response or "Error Message" in overview_response:
            raise ValueError(f"API error: {overview_response.get('Note') or overview_response.get('Error Message')}")

        # Get daily historical data (1 year)
        hist_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        hist_response = requests.get(hist_url).json()
        if "Time Series (Daily)" not in hist_response:
            raise ValueError("No historical data available")

        # Convert history to DataFrame
        hist_data = hist_response["Time Series (Daily)"]
        hist_df = pd.DataFrame.from_dict(hist_data, orient='index', dtype=float)
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df = hist_df.sort_index()  # Sort ascending
        hist_df = hist_df.tail(252)  # Approx 1 year (252 trading days)
        hist_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Standardize columns

        # Get news (using NEWS_SENTIMENT; note: limited to recent, may require premium for full)
        news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={ALPHA_VANTAGE_API_KEY}"
        news_response = requests.get(news_url).json()
        news = news_response.get("feed", [])[:3]  # Top 3 items

        data = {"info": overview_response, "hist": hist_df, "news": news}
        cache[ticker] = {"data": data, "timestamp": now}
        return cache[ticker]
    except Exception as e:
        err_str = str(e).lower()
        print(f"Error fetching data for {ticker}: {err_str}")  # Log error for debugging
        if any(keyword in err_str for keyword in ["rate limit", "too many requests", "429", "call frequency"]):
            raise  # Rethrow for retry
        else:
            raise ValueError(f"Non-recoverable error: {str(e)}")

def market_node(state):
    query = state["messages"][-1].content
    ticker_prompt = f"Extract the main stock ticker symbol from the query. If none, respond 'NONE'. Query: {query}"
    ticker_response = llm.invoke(ticker_prompt).content.strip().upper()
    if ticker_response == 'NONE':
        summary = "No valid stock ticker found in the query. Please specify a ticker for market analysis."
        return {"messages": [AIMessage(content=summary)]}

    try:
        cache_entry = get_stock_data(ticker_response)  # Now with enhanced retries
        data = cache_entry["data"]
        analysis_date = cache_entry["timestamp"].date()
        info = data["info"]
        hist = data["hist"]
        close = hist['Close']
        volume = hist['Volume']

        # RSI calculation
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Moving averages
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]

        # Volatility (std dev of returns)
        returns = close.pct_change()
        volatility = returns.std() * (252 ** 0.5)  # Annualized

        # Recent performance
        perf_1w = returns[-5:].mean() * 5 * 100 if len(returns) >= 5 else None
        perf_1m = returns[-21:].mean() * 21 * 100 if len(returns) >= 21 else None
        perf_3m = returns[-63:].mean() * 63 * 100 if len(returns) >= 63 else None

        perf_1w_str = f"{perf_1w:.2f}%" if perf_1w is not None else "N/A"
        perf_1m_str = f"{perf_1m:.2f}%" if perf_1m is not None else "N/A"
        perf_3m_str = f"{perf_3m:.2f}%" if perf_3m is not None else "N/A"

        # Signals
        signals = []
        if rsi < 30: signals.append("Oversold (potential buy)")
        if rsi > 70: signals.append("Overbought (potential sell)")
        if price > ma20 > ma50: signals.append("Bullish trend")
        if price < ma20 < ma50: signals.append("Bearish trend")
        if ma50 > ma200: signals.append("Long-term uptrend")
        if ma50 < ma200: signals.append("Long-term downtrend")

        # Volume trend
        avg_volume = volume.rolling(20).mean().iloc[-1]
        recent_volume = volume.iloc[-1]
        volume_signal = "Increasing volume (potential momentum)" if recent_volume > avg_volume else "Decreasing volume"

        # News insights (recent headlines) - Adapted for Alpha Vantage format
        news = data["news"]  # List of dicts from Alpha Vantage
        news_summary = "\n".join([f"- {n.get('title', 'N/A')} ({n.get('authors', 'N/A')}, {n.get('url', 'N/A')})" for n in news]) if news else "No recent news"

        # Overall rating
        buy_signals = sum(1 for s in signals if "buy" in s.lower() or "bullish" in s.lower() or "uptrend" in s.lower())
        sell_signals = sum(1 for s in signals if "sell" in s.lower() or "bearish" in s.lower() or "downtrend" in s.lower())
        rating = "STRONG BUY" if buy_signals > sell_signals + 1 else "BUY" if buy_signals > sell_signals else "STRONG SELL" if sell_signals > buy_signals + 1 else "SELL" if sell_signals > buy_signals else "HOLD"

        # Handle fundamentals formatting - Adapted for Alpha Vantage fields
        pe = info.get('ForwardPE')
        pe_str = f"{pe:.1f}" if isinstance(pe, (int, float)) else 'N/A'

        tp = info.get('AnalystTargetPrice')
        tp_str = f"${tp:.2f}" if isinstance(tp, (int, float)) else 'N/A'

        mc = info.get('MarketCapitalization')
        mc_str = f"${float(mc) / 1e9:.2f}B" if mc else 'N/A'

        div_yield = info.get('DividendYield')
        div_yield_str = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) else 'N/A'

        summary = f"""
**{ticker_response} Market Analysis (as of {analysis_date})**

**Current Metrics:**
- Current Price: ${price:.2f}
- 20-Day MA: ${ma20:.2f}
- 50-Day MA: ${ma50:.2f}
- 200-Day MA: ${ma200:.2f}
- RSI (14): {rsi:.1f}
- Annualized Volatility: {volatility:.2%}

**Performance:**
- 1-Week Return: {perf_1w_str}
- 1-Month Return: {perf_1m_str}
- 3-Month Return: {perf_3m_str}

**Trend Signals:**
- {', '.join(signals) if signals else 'Neutral'}
- {volume_signal}

**Fundamentals:**
- Forward P/E: {pe_str}
- Mean Target Price: {tp_str}
- Market Cap: {mc_str}
- Dividend Yield: {div_yield_str}

**Recent News Insights:**
{news_summary}

**Overall Recommendation:** {rating}
**Insights:** The stock shows {'positive momentum' if buy_signals > sell_signals else 'caution advised' if sell_signals > buy_signals else 'stable conditions'}. Monitor for breakout above resistance levels.
        """
    except Exception as e:
        summary = f"Error analyzing {ticker_response}: {str(e)}. Please try again later or check the ticker symbol."

    return {"messages": [AIMessage(content=summary)]}