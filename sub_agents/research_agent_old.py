import sys
import os
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
from ..agent_utils import return_distribution_analysis, DATA_CACHE

# --- MEDIATOR TOOL ---
def _analyze_by_id(reference_id: str) -> dict:
    """Reads cache by ID, runs math analysis, returns STATS dict."""
    data = DATA_CACHE.get(reference_id)
    if not data: return {"status": "error", "msg": "ID not found"}
    
    return return_distribution_analysis({"data":data})

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

research_agent = LlmAgent(
    name="research_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True,thinking_budget=8192)),
    generate_content_config=GenerateContentConfig(temperature=0.0),
    
    tools=[_analyze_by_id],

    instruction="""
You are an expert Quantitative Researcher. Your goal is to propose a trading strategy based on statistical distribution analysis.

**INPUT:**
You will receive a `reference_id` (UUID string) representing a dataset stored in memory.

**WORKFLOW:**
1. Call the tool `run_analysis_on_cached_data(reference_id=...)`.
2. Analyze the returned JSON statistics for both Short-Term (60d) and Long-Term (1y) horizons.

**ANALYSIS LOGIC:**
- **Kurtosis:**
    - High (> 3.0): Indicates "Fat Tails". Price tends to have extreme moves followed by reversion. **Strategy:** Mean Reversion, Bollinger Band fades, or Short Volatility (Iron Condors).
    - Low (< 3.0): Normal distribution. **Strategy:** Standard trend following.
- **Skewness:**
    - Positive (> 0): Small frequent losses, large occasional gains. **Strategy:** Momentum / Trend Following (Breakouts).
    - Negative (< 0): Small frequent gains, large occasional crashes. **Strategy:** Defensive, Buy Puts for protection.
- **Volatility:**
    - If Short-Term Volatility < Long-Term Volatility: Market is compressing. **Strategy:** Prepare for Breakout (Straddles).
    - If Short-Term Volatility > Long-Term Volatility: Market is extended. **Strategy:** Reduce Position Sizing.

**OUTPUT FORMAT:**
You must output a structured text report:
1.  **Market Regime:** (e.g., "High Volatility / Mean Reverting" or "Stable / Trending")
2.  **Key Metrics:** List the specific Skew/Kurtosis/Vol values you relied on.
3.  **Proposed Strategy:** Define the entry/exit logic based on the metrics.
    """,
    output_key="researcher_response",
)

root_agent = research_agent
