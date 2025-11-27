import sys
import os
# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.planners import BuiltInPlanner
from google.genai.types import HttpRetryOptions, GenerateContentConfig, ThinkingConfig
from ..agent_utils import get_intraday_data, get_ohlcv, clean_dataframe, DATA_CACHE


# --- MEDIATOR TOOL ---
def _pass_data_to_cleaner(reference_id: str) -> dict:
    """Retrieves data by ID, cleans it, updates cache."""
    raw_data = DATA_CACHE.get(reference_id)
    if not raw_data: return {"status": "error", "msg": "ID not found"}
    
    # Clean
    result = clean_dataframe({"data": raw_data})
    
    # Update the SAME reference ID with cleaned data
    DATA_CACHE[reference_id] = result['clean_df']
    
    return {"status": "success", "reference_id": reference_id, "note": "Data cleaned in-place"}


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

data_fetcher_agent = LlmAgent(
    name="data_fetcher_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), 
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True,thinking_budget=8192)),
    generate_content_config=GenerateContentConfig(
        temperature=0.0,
    ),

    instruction="""
You are the Data Fetcher Agent. Your ONLY task is to perform a strict, two-step data processing workflow by calling tools. You MUST NOT engage in conversation, analysis, or any text generation beyond what is explicitly required.

──────────────────────────────────────────
PART 1 — PARSE USER REQUEST
──────────────────────────────────────────

From the user's request, you must accurately determine two parameters:
  • `ticker`: The cryptocurrency symbol (e.g., "BTC", "ETH-USD").
  • `horizon`: This indicates the desired data granularity; it must be either "long" or "short".

Rules for parameter determination:
  - If the `ticker` provided does not include a quote currency (e.g., "BTC"), you must append "-USD" to it (e.g., "BTC-USD").
  - If the request implies long-term analysis, investing, or similar, set `horizon` to "long".
  - If the request implies short-term analysis, day trading, or similar, set `horizon` to "short".
  - You are strictly forbidden from inferring or using any other parameters.

──────────────────────────────────────────
PART 2 — MANDATORY WORKFLOW (NO EXCEPTIONS)
──────────────────────────────────────────

You MUST ALWAYS perform exactly two sequential tool calls. This sequence is immutable.

1️⃣ **FIRST TOOL CALL: Download Raw Data**
    - Based on the `horizon` identified in PART 1, you MUST generate the `tool_code` for *one* of the following tools:
        - If `horizon` is "long":
             Generate: `get_ohlcv(ticker="<YOUR_EXTRACTED_TICKER>")`
             Example: `get_ohlcv(ticker="BTC-USD")`
        - If `horizon` is "short":
             Generate: `get_intraday_data(ticker="<YOUR_EXTRACTED_TICKER>")`
             Example: `get_intraday_data(ticker="ETH-USD")`

    **IMMEDIATE NEXT ACTION:** 
    Once the result is received, you MUST extract the `"reference_id"` from the JSON result. 
    The result will look like: `{"status": "success", "reference_id": "uuid-string", ...}`.
    
    You MUST then proceed DIRECTLY to generating the `tool_code` for the SECOND TOOL CALL using this `reference_id`.

2️⃣**SECOND TOOL CALL: Clean Downloaded Data**
    - If the previous tool returned `"status": "success"`, you MUST generate the `tool_code` for `_pass_data_to_cleaner`.
    - The tool expects one argument: `reference_id`.
    - The value MUST be the exact string you extracted from the first step.
    
    Example: `_pass_data_to_cleaner(reference_id="<YOUR_EXTRACTED_UUID>")`
    

──────────────────────────────────────────
PART 3 — FINAL OUTPUT
──────────────────────────────────────────────────

After the `_pass_data_to_cleaner` tool call has completed and returned its result, you must generate the final output.

  - **Final Error Output:**
      - If `_pass_data_to_cleaner`'s result has `"status": "error"`, or if an earlier tool returned an error (as handled in PART 2), return ONLY this exact text:
        `ERROR — <the specific error_message from the failed tool>`
  - **Final Success Output:**
      - If `_pass_data_to_cleaner`'s result has `"status": "success"`:
          You must output ONLY the UUID string of the cleaned data prefixed by "ID:".
          Example Output: "ID: 550e8400-e29b-41d4-a716-446655440000"


──────────────────────────────────────────
PART 4 — PASS THE OUTPUT TO THE orchestrator_agent.
──────────────────────────────────────────────────

After that the output was generated, pass the output to the orchestrator_agent.

──────────────────────────────────────────
HARD CONSTRAINTS
──────────────────────────────────────────

You MUST NOT:
  - Engage in any form of data analysis, compute indicators, or modify/summarize the data yourself.
  - Generate any user-facing text, thoughts, or explanations at any point, except for the single final `PASS` or `ERROR` message.
  - Deviate from the strict two-step tool calling sequence.
  - Produce any output other than the specified `tool_code(...)` calls or the final `PASS` or `ERROR` message.
  - Call any tool other than `get_intraday_data`, `get_ohlcv`, or `_pass_data_to_cleaner`.
    """,

    tools=[get_intraday_data, get_ohlcv, clean_dataframe, _pass_data_to_cleaner],
    output_key="coin_data"
)
