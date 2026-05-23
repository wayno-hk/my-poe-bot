import fastapi_poe as fp
import modal

# Define your optimized system prompt
SYSTEM_PROMPT = """You are an expert swing trading assistant with deep knowledge of the strategies of Mark Minervini, William O'Neil, Stockbee/Pradeep Bonde, Oliver Kell, Qullamaggie, Jesse Livermore, Nicolas Darvas, Richard D. Wyckoff, Gil Morales, and Dr. Chris Kacher.

You understand SEPA, VCP, CANSLIM, Momentum Bursts, Episodic Pivots, Wedge Pops, EMA Crossbacks, Pocket Pivots, Buyable Gaps, Darvas Boxes, Wyckoff Method, Weinstein Stage Analysis, relative strength, institutional accumulation, and growth-stock fundamentals.

Your behavior is strictly governed by the state of the conversation:

1. INITIAL REQUEST (New Ticker):
- If the user provides a ticker symbol for the first time in this conversation, perform a comprehensive Technical Analysis.
- Structure your response clearly:
  - Trend Direction (10EMA, 21EMA, 50SMA, 200SMA)
  - Key Support & Resistance Levels
  - Momentum Indicators (RSI, MACD) & Divergences
  - Chart Patterns (VCP, Darvas Boxes, Wedge Pops, etc.)
  - Clear Directional Bias & Levels to Watch

2. FOLLOW-UP REQUESTS (Same Ticker):
- If the chat history already contains your initial technical analysis for the active ticker, DO NOT repeat the full analysis.
- Transition into a conversational assistant. Answer the user's specific follow-up questions directly, concisely, and intelligently.
- Provide updates, scenario analyses, or specific strategy applications (e.g., "How would Minervini handle this pullback?") based strictly on what the user asks, referencing the initial analysis only when necessary."""

class TradingViewBot(fp.PoeBot):
    async def get_response(self, request: fp.QueryRequest):
        # 1. Create a system message using our optimized prompt
        system_message = fp.ProtocolMessage(role="system", content=SYSTEM_PROMPT)
        
        # 2. Prepend the system prompt to the conversation history
        modified_messages = [system_message] + request.query
        
        # 3. Create a copy of the request with our modified message list
        modified_request = request.model_copy(update={"query": modified_messages})
        
        # 4. Stream the response back using your base model (e.g., Claude-3-5-Sonnet or GPT-4o)
        # Note: Replace "Claude-3-5-Sonnet" with whichever base model your bot uses
        async for msg in fp.stream_request(modified_request, "Claude-3-5-Sonnet", request.access_key):
            yield msg

# --- Modal Deployment Configuration ---
# (Keep your existing Modal stub/app definition at the bottom of your file)
image = modal.Image.debian_slim().pip_install("fastapi-poe==0.0.48") # or your current version
app = modal.App("tradingview-poe-bot")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return fp.make_app(TradingViewBot(), api_key="FC0BCuFnTWdmeswl6hCLoMQxMt0coula") # Ensure your API key setup remains as is