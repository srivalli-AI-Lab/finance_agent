# agents/portfolio_agent.py
import requests  # For API calls
import pandas as pd
from langchain_core.messages import AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env file")

# Use a global dict for caching with TTL
cache = {}  # {symbol: {"data": {"price": ..., "sector": ...}, "timestamp": datetime}}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def get_symbol_info(symbol):
    now = datetime.now()
    if symbol in cache and now - cache[symbol]["timestamp"] < timedelta(minutes=15):
        return cache[symbol]["data"]

    try:
        # Get price from GLOBAL_QUOTE
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        quote_response = requests.get(quote_url).json()
        quote_data = quote_response.get("Global Quote", {})
        if not quote_data:
            raise ValueError(f"No quote data for {symbol}")

        price = float(quote_data.get("05. price", 0))

        # Get sector from OVERVIEW
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        overview_response = requests.get(overview_url).json()
        if "Note" in overview_response or "Error Message" in overview_response:
            raise ValueError(f"API error: {overview_response.get('Note') or overview_response.get('Error Message')}")

        sector = overview_response.get("Sector", "Unknown")

        data = {"price": price, "sector": sector}
        cache[symbol] = {"data": data, "timestamp": now}
        return data
    except Exception as e:
        err_str = str(e).lower()
        print(f"Error fetching info for {symbol}: {err_str}")
        if any(keyword in err_str for keyword in ["rate limit", "too many requests", "429", "call frequency"]):
            raise  # Rethrow for retry
        return {"price": 0, "sector": "Error"}

def portfolio_node(state):
    df = state["df"]
    if df is None:
        return {"messages": [AIMessage(content="Upload Excel with 'Symbol' and 'Quantity' first!")]}

    symbols = df['Symbol'].dropna().unique()
    prices, sectors = {}, {}
    for s in symbols:
        try:
            info = get_symbol_info(s)
            prices[s] = info["price"]
            sectors[s] = info["sector"]
            time.sleep(1)  # Add small delay between symbols to respect rate limits
        except:
            prices[s] = 0
            sectors[s] = 'Error'

    df['Price'] = df['Symbol'].map(prices).fillna(0)
    df['Value'] = df['Quantity'] * df['Price']
    total = df['Value'].sum()
    if total == 0:
        return {"messages": [AIMessage(content="No valid data. Check symbols.")]}

    df['Weight_%'] = (df['Value'] / total * 100).round(2)
    df['Sector'] = df['Symbol'].map(sectors)
    sector_alloc = df.groupby('Sector')['Weight_%'].sum().round(2)

    max_weight = df['Weight_%'].max()
    risk = "Low" if max_weight < 10 and len(df) > 8 else "High" if max_weight > 20 else "Medium"

    suggestions = []
    if max_weight > 10: suggestions.append(f"Reduce {df.loc[df['Weight_%'].idxmax(), 'Symbol']} ({max_weight}%)")
    if len(sector_alloc) < 3: suggestions.append("Add 2+ sector ETFs")
    if len(df) < 5: suggestions.append("Add VTI (broad market)")

    table = df[['Symbol', 'Quantity', 'Price', 'Value', 'Weight_%']].round(2).to_markdown(index=False)

    response = f"""
**Portfolio** | Total: **${total:,.0f}**

**Holdings**
{table}

**Sectors**
{sector_alloc.to_markdown()}

**Risk**: {risk}

**Suggestions**
""" + "\n".join([f"- {s}" for s in suggestions]) + "\n\nRebalance quarterly!"
    return {"messages": [AIMessage(content=response)]}