import re

from google.adk.agents import LlmAgent,SequentialAgent,Agent
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

# Import the sub-agents
from .sub_agents.coin_verification_agent import coin_verification_agent
from .sub_agents.data_fetcher_agent import data_fetcher_agent
from .sub_agents.research_agent import research_agent



orchestrator_agent = Agent(
    name="orchestrator_agent",
    model=Gemini(model="gemini-2.5-flash"),
    planner=BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True,thinking_budget=8192)),
    sub_agents=[coin_verification_agent, data_fetcher_agent, research_agent],
    instruction="""
    You are the Strategy Orchestrator.
    
    MANDATORY WORKFLOW:
    
    1. EXTRACT the ticker symbol (e.g., BTC, ETH) and horizon from user input.
    
    2. CALL coin_verification_agent.
       - IF output is "REJECT": STOP immediately and inform the user the coin is invalid.
       - IF output is "PASS": Proceed to next step (step 3).
       
    3. CALL `data_fetcher_agent`.
       - This returns a Reference ID (UUID).
       - If it returns "ERROR": STOP.
       
    4. CALL `research_agent`.
       - Present the strategy to the user.
    """
)


root_agent = orchestrator_agent
