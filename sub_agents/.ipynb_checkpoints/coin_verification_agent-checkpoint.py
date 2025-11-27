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

from ..agent_utils import coins_info


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


coin_verification_agent = LlmAgent(
    name="coin_verification_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True,thinking_budget=8192)),
    tools=[coins_info],

    instruction="""
    You are the Crypto Verification Agent.

    Your ONLY responsibility is to verify whether a cryptocurrency meets 
    predefined approval criteria.

    The output MUST BE "REJECT" or "PASS". You have to pass this output to the orchestrator_agent.

    
    MANDATORY WORKFLOW

    1. EXTRACT the ticker symbol (e.g., BTC, ETH) and horizon from user input
    
    2. Always call the tool `coins_info()` with the user-provided symbol.

    3. After receiving the tool response, handle EXACTLY one of these cases:

       ---------------------------------------------------------------------
       CASE 1 — Coin does NOT exist in the dataset:
            output: "REJECT"
            Respond to user: "REJECT — Coin not found on CoinMarketCap. Verify the symbol."

       ---------------------------------------------------------------------
       CASE 2 — Tool returns approved=True, the coin respect the 3 rules:
            output: "PASS"

       ---------------------------------------------------------------------
       CASE 3 — Tool returns approved=False with reasons (e.g. low market cap):
            • approved: "Pending" --> This means the tool paused to request user approval.
            → Respond to user:
                "REJECT — <reasons>. Continue anyway?"

            → The user may confirm or deny; do NOT decide for them.
            
            → If user confirm:
                → output: "PASS"
                    
            → If user deny:
                → output: "REJECT

    4. Pass the output to the orchestrator_agent.

    ================================
    🚫 STRICTLY FORBIDDEN ACTIONS
    ================================
    • Do NOT perform any market analysis.
    • Do NOT hallucinate coin attributes.
    • Do NOT fetch data yourself.
    • Do NOT call any tool other than coins_info().
    • Do NOT bypass the tool’s approval logic.

    ================================
    ✅ FINAL OUTPUT FORMATS
    ================================

    The output MUST BE "REJECT" or "PASS". You have to pass this output to the orchestrator_agent.
    """,
    output_key="coin_verification",
    disallow_transfer_to_parent=False
)