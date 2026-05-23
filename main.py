import re
import modal
import copy
from fastapi_poe import PoeBot, make_app
from fastapi_poe.types import QueryRequest, SettingsRequest, SettingsResponse, ProtocolMessage
from fastapi_poe.client import stream_request
from tradingview_ta import TA_Handler, Interval

# --- CONFIGURATION: SET YOUR PREFERRED GEMINI MODEL HERE ---
GEMINI_MODEL_ID = "Gemini-3.1-Pro"

# Define your optimized system prompt template
SYSTEM_PROMPT_TEMPLATE = """You are an expert swing trading assistant with deep knowledge of the strategies of Mark Minervini, William O'Neil, Stockbee/Pradeep Bonde, Oliver Kell, Qullamaggie, Jesse Livermore, Nicolas Darvas, Richard D. Wyckoff, Gil Morales, and Dr. Chris Kacher.

You understand SEPA, VCP, CANSLIM, Momentum Bursts, Episodic Pivots, Wedge Pops, EMA Crossbacks, Pocket Pivots, Buyable Gaps, Darvas Boxes, Wyckoff Method, Weinstein Stage Analysis, relative strength, institutional accumulation, and growth-stock fundamentals.

Your behavior is strictly governed by the state of the conversation:

1. INITIAL REQUEST (New Ticker):
- If the user provides a ticker symbol for the first time in this conversation, perform a comprehensive Technical Analysis using the TradingView indicator data provided below.
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

--- TRADINGVIEW DATA FOR THE ACTIVE TICKER ({ticker}) ---
{tradingview_data}
"""

def fetch_tradingview_data(ticker: str) -> str:
    """Fetches real-time technical analysis indicators from TradingView."""
    try:
        # Clean ticker name (remove $ sign if present)
        clean_ticker = ticker.strip().upper().replace("$", "")
        
        # We try NASDAQ first, then NYSE as a fallback
        exchanges = ["NASDAQ", "NYSE", "AMEX"]
        analysis = None
        
        for exchange in exchanges:
            try:
                handler = TA_Handler(
                    symbol=clean_ticker,
                    screener="america",
                    exchange=exchange,
                    interval=Interval.INTERVAL_1_DAY,
                    timeout=5
                )
                analysis = handler.get_analysis()
                if analysis:
                    break
            except Exception:
                continue
                
        if not analysis:
            return "Could not retrieve live TradingView indicators for this ticker. Proceed with general knowledge."
            
        indicators = analysis.indicators
        summary = analysis.summary
        
        # Format a clean string of the key indicators for the AI
        data_summary = f"""
- Price (Close): {indicators.get('close')}
- Open: {indicators.get('open')} | High: {indicators.get('high')} | Low: {indicators.get('low')}
- Recommendation Summary: {summary.get('RECOMMENDATION')} (Buy: {summary.get('BUY')}, Sell: {summary.get('SELL')}, Neutral: {summary.get('NEUTRAL')})
- Exponential Moving Averages: EMA10={indicators.get('EMA10')}, EMA20={indicators.get('EMA20')}, EMA50={indicators.get('EMA50')}, EMA100={indicators.get('EMA100')}, EMA200={indicators.get('EMA200')}
- Simple Moving Averages: SMA10={indicators.get('SMA10')}, SMA20={indicators.get('SMA20')}, SMA50={indicators.get('SMA50')}, SMA100={indicators.get('SMA100')}, SMA200={indicators.get('SMA200')}
- Relative Strength Index (RSI 14): {indicators.get('RSI')}
- MACD: Line={indicators.get('MACD.macd')}, Signal={indicators.get('MACD.signal')}
- Average True Range (ATR): {indicators.get('ATR')}
- Bollinger Bands: Upper={indicators.get('BB.upper')}, Lower={indicators.get('BB.lower')}
- Volume: {indicators.get('volume')}
"""
        return data_summary
    except Exception as e:
        return f"Error fetching TradingView data: {str(e)}"

def extract_ticker(text: str) -> str:
    """Extracts a stock ticker symbol from a user message (e.g., $AAPL, AAPL, NVDA)."""
    # Look for standard ticker patterns (1-5 letters, optionally preceded by $)
    match = re.search(r'\$?([A-Z]{1,5})\b', text.upper())
    if match:
        return match.group(1)
    return ""

class TradingViewBot(PoeBot):
    async def get_response(self, request: QueryRequest):
        # 1. Analyze conversation history to determine if this is a follow-up
        user_messages = [msg for msg in request.query if msg.role == "user"]
        bot_messages = [msg for msg in request.query if msg.role == "bot"]
        
        current_user_message = user_messages[-1].content if user_messages else ""
        ticker = extract_ticker(current_user_message)
        
        # If no ticker was found in the current message, look back at previous messages
        if not ticker:
            for msg in reversed(user_messages):
                ticker = extract_ticker(msg.content)
                if ticker:
                    break
                    
        # 2. Fetch live data if we found a ticker, otherwise use a placeholder
        tradingview_data = ""
        if ticker:
            tradingview_data = fetch_tradingview_data(ticker)
        else:
            ticker = "Unknown Ticker"
            tradingview_data = "No ticker detected in the conversation yet."

        # 3. Format the system prompt with our fetched data
        system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
            ticker=ticker,
            tradingview_data=tradingview_data
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
image = modal.Image.debian_slim().pip_install("fastapi-poe==0.0.48", "tradingview-ta")
stub = modal.Volume.from_name("poe-bot-vol", create_if_missing=True)
app = modal.App("tradingview-poe-bot")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    # Your Poe API Key
    return make_app(TradingViewBot(), api_key="FC0BCuFnTWdmeswl6hCLoMQxMt0coula")