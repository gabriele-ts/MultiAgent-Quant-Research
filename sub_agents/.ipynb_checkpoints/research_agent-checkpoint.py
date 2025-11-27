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

from ..agent_utils import (
    return_distribution_analysis,
    autocorrelation_analysis,
    DATA_CACHE
)

# -------------------------------------------------------------------
# -----------------------  MEDIATOR TOOL  ---------------------------
# -------------------------------------------------------------------

def _analyze_by_id(reference_id: str) -> dict:
    """
    Reads cached dataset by ID, runs BOTH statistical analyses, 
    and returns a combined dict to the agent.
    """
    data = DATA_CACHE.get(reference_id)
    if not data:
        return {"status": "error", "msg": "ID not found"}
    
    dist_stats = return_distribution_analysis({"data": data})
    autocorr_stats = autocorrelation_analysis({"data": data})

    return {
        "status": "success",
        "distribution": dist_stats,
        "autocorrelation": autocorr_stats
    }

# -------------------------------------------------------------------
# ----------------------- RETRY / MODEL CONFIG ----------------------
# -------------------------------------------------------------------

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# -------------------------------------------------------------------
# -----------------------   THE LLM AGENT   -------------------------
# -------------------------------------------------------------------

research_agent = LlmAgent(
    name="research_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    planner=BuiltInPlanner(
        thinking_config=ThinkingConfig(
            include_thoughts=True,
            thinking_budget=8192
        )
    ),
    generate_content_config=GenerateContentConfig(
        temperature=0.0
    ),

    tools=[_analyze_by_id],

    instruction="""
You are an expert Quantitative Researcher. Your goal is to propose a trading strategy using 
both **statistical distribution analysis** and **autocorrelation structure**.

---------------------------------------------------
INPUT
---------------------------------------------------
You will receive a reference_id (UUID string) that maps to a dataset in memory.

---------------------------------------------------
WORKFLOW
---------------------------------------------------
1. Call the tool:
       run_analysis_on_cached_data(reference_id=...)
2. You will receive JSON with two blocks:
       - distribution: short- & long-term return distribution stats
       - autocorrelation: lag-based autocorrelation metrics
3. Use BOTH to determine:
       - Market Regime
       - Dominant statistical behaviors
       - Appropriate trading approach

---------------------------------------------------
ANALYSIS LOGIC
---------------------------------------------------

### 1) RETURN DISTRIBUTION
- **Kurtosis > 3** → Fat Tails, Mean Reversion, Short Volatility structures.
- **Kurtosis < 3** → Normal-like, Trend Following appropriate.

- **Skewness > 0** → Upside tail risk, Momentum Bias.
- **Skewness < 0** → Downside crashes, Defensive hedging required.

- **Volatility Regime**
    - Short-Term Vol < Long-Term Vol → Vol compression → Breakout prep.
    - Short-Term Vol > Long-Term Vol → Extended move → Reduce sizing.

### 2) AUTOCORRELATION INTERPRETATION
- **Positive autocorrelation at lags 1–5**
    → Short-term Momentum / Trend Persistence.

- **Negative autocorrelation at lags 1–5**
    → Short-term Mean Reversion / Oscillatory regime.

- **ACF close to zero for all lags**
    → Market is memoryless → Purely stochastic / Random Walk.

- **Significance Threshold**
    Compare autocorrelation values to "approx_95pct_conf_band":
    - |ACF(lag)| > conf_band → Statistically meaningful
    - |ACF(lag)| < conf_band → Noise / Not reliable

### 3) CROSS-INTERPRETATION
Use both analyses together:
- Positive Skew + Positive ACF → Strong trend following.
- Fat Tails + Negative ACF → Violent mean reversion → Fade breakouts.
- Volatility compression + ACF trending positive → Impending trend expansion.

---------------------------------------------------
OUTPUT FORMAT
---------------------------------------------------

Your output MUST be a structured text report with:

1. **Market Regime**  
   e.g., "High Volatility Mean-Reversion", "Low Volatility Trend-Building", etc.

2. **Key Metrics Used**  
   List the EXACT skew/kurtosis/volatility/autocorrelation signals relied upon.

3. **Proposed Strategy**  
   Provide:
   - Theoretical justification
   - Entry logic
   - Exit logic
   - Risk management (vol-adjusted sizing, hedging, stop levels)

Always ensure the final strategy reflects BOTH distribution AND autocorrelation statistics.
    """,

    output_key="researcher_response",
)

# Root entrypoint
root_agent = research_agent
