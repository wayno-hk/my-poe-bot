import re
import modal
import copy
from fastapi_poe import PoeBot, make_app
from fastapi_poe.types import QueryRequest, SettingsRequest, SettingsResponse, ProtocolMessage
from fastapi_poe.client import stream_request

# --- CONFIGURATION: SET YOUR PREFERRED GEMINI MODEL HERE ---
GEMINI_MODEL_ID = "Claude-Opus-4.6"

# Define your optimized system prompt template
SYSTEM_PROMPT_TEMPLATE = """You are an expert swing trading assistant with deep knowledge of the strategies of Mark Minervini, William O'Neil, Stockbee/Pradeep Bonde, Oliver Kell, Qullamaggie, Jesse Livermore, Nicolas Darvas, Richard D. Wyckoff, Gil Morales, and Dr. Chris Kacher.

You understand SEPA, VCP, CANSLIM, Momentum Bursts, Episodic Pivots, Wedge Pops, EMA Crossbacks, Pocket Pivots, Buyable Gaps, Darvas Boxes, Wyckoff Method, Weinstein Stage Analysis, relative strength, institutional accumulation, and growth-stock fundamentals.

Your behavior is strictly governed by the state of the conversation:

1. INITIAL REQUEST (New Ticker):
- If the user provides a ticker symbol for the first time in this conversation, perform a comprehensive Technical Analysis using the Yahoo Finance indicator data provided below.
- Structure your response clearly:
  - Trend Direction (10EMA, 21EMA, 50SMA, 200SMA)
  - Key Support & Resistance Levels
  - Momentum Indicators (RSI, MACD) & Divergences
  - Chart Patterns (VCP, Darvas Boxes, Wedge Pops, etc.)
  - Clear Directional Bias & Levels to Watch

2. FOLLOW-UP REQUESTS (Same Ticker):
- If the chat history already contains your initial technical analysis for the active ticker, DO NOT repeat the full analysis.
- Transition into a conversational assistant. Answer the user's specific follow-up questions directly, concisely, and intelligently.
- Provide updates, scenario analyses, or specific strategy applications (e.g., "How would Minervini handle this pullback?") based strictly on what the user asks, referencing the initial analysis only when necessary.

--- YAHOO FINANCE DATA FOR THE ACTIVE TICKER ({ticker}) ---
{yfinance_data}
"""

def fetch_yfinance_data(ticker: str) -> str:
    """Fetches real-time technical analysis indicators from Yahoo Finance."""
    # Move imports here so they are only loaded inside the remote container
    import yfinance as yf
    import pandas as pd
    
    try:
        # Clean ticker name (remove $ sign if present)
        clean_ticker = ticker.strip().upper().replace("$", "")
        
        t = yf.Ticker(clean_ticker)
        # Fetch 1 year of daily data to calculate EMA200/SMA200 accurately
        df = t.history(period="1y", interval="1d")
        
        if df.empty:
            return "Could not retrieve live Yahoo Finance data for this ticker. Proceed with general knowledge."
            
        # Extract latest prices
        close_val = df['Close'].iloc[-1]
        open_val = df['Open'].iloc[-1]
        high_val = df['High'].iloc[-1]
        low_val = df['Low'].iloc[-1]
        volume_val = df['Volume'].iloc[-1]
        
        # Exponential Moving Averages
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Simple Moving Averages
        df['SMA10'] = df['Close'].rolling(window=10).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Relative Strength Index (RSI 14)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / ema_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_line'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        
        # Average True Range (ATR 14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        df['BB_mid'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)
        
        # Helper function to round values safely
        def r(val):
            return round(val, 2) if pd.notna(val) else "N/A"
            
        # Format a clean string of the key indicators for the AI
        data_summary = f"""
- Price (Close): {r(close_val)}
- Open: {r(open_val)} | High: {r(high_val)} | Low: {r(low_val)}
- Exponential Moving Averages: EMA10={r(df['EMA10'].iloc[-1])}, EMA20={r(df['EMA20'].iloc[-1])}, EMA50={r(df['EMA50'].iloc[-1])}, EMA100={r(df['EMA100'].iloc[-1])}, EMA200={r(df['EMA200'].iloc[-1])}
- Simple Moving Averages: SMA10={r(df['SMA10'].iloc[-1])}, SMA20={r(df['SMA20'].iloc[-1])}, SMA50={r(df['SMA50'].iloc[-1])}, SMA100={r(df['SMA100'].iloc[-1])}, SMA200={r(df['SMA200'].iloc[-1])}
- Relative Strength Index (RSI 14): {r(df['RSI'].iloc[-1])}
- MACD: Line={r(df['MACD_line'].iloc[-1])}, Signal={r(df['MACD_signal'].iloc[-1])}
- Average True Range (ATR): {r(df['ATR'].iloc[-1])}
- Bollinger Bands: Upper={r(df['BB_upper'].iloc[-1])}, Lower={r(df['BB_lower'].iloc[-1])}
- Volume: {int(volume_val) if pd.notna(volume_val) else "N/A"}
"""
        return data_summary
    except Exception as e:
        return f"Error fetching Yahoo Finance data: {str(e)}"

def extract_ticker(text: str) -> str:
    """Extracts a stock ticker symbol from a user message (e.g., $AAPL, AAPL, NVDA)."""
    # Look for standard ticker patterns (1-5 letters, optionally preceded by $)
    match = re.search(r'\$?([A-Z]{1,5})\b', text.upper())
    if match:
        return match.group(1)
    return ""

class YahooFinanceBot(PoeBot):
    async def get_response(self, request: QueryRequest):
        # 1. Analyze conversation history to determine if this is a follow-up
        user_messages = [msg for msg in request.query if msg.role == "user"]
        
        current_user_message = user_messages[-1].content if user_messages else ""
        ticker = extract_ticker(current_user_message)
        
        # If no ticker was found in the current message, look back at previous messages
        if not ticker:
            for msg in reversed(user_messages):
                ticker = extract_ticker(msg.content)
                if ticker:
                    break
                    
        # 2. Fetch live data if we found a ticker, otherwise use a placeholder
        yfinance_data = ""
        if ticker:
            yfinance_data = fetch_yfinance_data(ticker)
        else:
            ticker = "Unknown Ticker"
            yfinance_data = "No ticker detected in the conversation yet."

        # 3. Format the system prompt with our fetched data
        system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
            ticker=ticker,
            yfinance_data=yfinance_data
        )
        
        system_message = ProtocolMessage(role="system", content=system_prompt_content)
        
        # 4. Prepend the system prompt to the conversation history
        modified_messages = [system_message] + request.query
        
        # 5. Create a copy of the request with our modified message list
        modified_request = request.model_copy(update={"query": modified_messages})
        
        # 6. Stream the response back using your preferred Gemini model
        async for msg in stream_request(modified_request, GEMINI_MODEL_ID, request.access_key):
            yield msg

    # --- DECLARE DEPENDENCY TO PREVENT "UNEXPECTED ISSUE" ERROR ---
    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        return SettingsResponse(
            server_bot_dependencies={GEMINI_MODEL_ID: 1}
        )

# --- Modal Deployment Configuration ---
image = modal.Image.debian_slim().pip_install(
    "fastapi-poe==0.0.48", 
    "yfinance", 
    "pandas", 
    "lxml"
)
stub = modal.Volume.from_name("poe-bot-vol", create_if_missing=True)
app = modal.App("tradingview-poe-bot")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    # Your Poe API Key
    return make_app(YahooFinanceBot(), api_key="FC0BCuFnTWdmeswl6hCLoMQxMt0coula")