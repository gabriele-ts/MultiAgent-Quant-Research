# MultiAgent Quant Research

A modular multi-agent system designed for quantitative research, market data analysis, and trading strategy generation. Built using OpenAI's ADK framework, it integrates multiple specialized sub‑agents to process financial data, clean and normalize OHLCV datasets, and provide insights for developing trading strategies.

## 🚀 Features
- **Multi-agent architecture** optimized for quant workflows
- **Data cleaning & normalization** (OHLCV, intraday, missing data handling)
- **Financial data analysis** using specialized sub‑agents
- **Strategy suggestion engine** powered by LLM reasoning
- **Extensible design** for adding new agents or tool functions

## 📁 Project Structure
```bash
MultiAgent-Quant-Research/
├── agent.py               # Main orchestrator agent
├── agent_utils.py         # Shared utility functions
├── sub_agents/            # Specialized agents (data, analysis, strategy)
├── __init__.py
└── README.md
```

## 🧠 How It Works
The system uses an orchestrator agent that delegates tasks to specialized sub‑agents:
- **Coin Verification Agent** - responsible for validating whether a cryptocurrency meets the project’s predefined approval criteria
- **Data Agent** — fetches raw OHLCV and intraday data
- **Cleaning Agent** — cleans, resamples, and fixes irregularities in datasets
- **Analysis Agent** — computes metrics, detects patterns, explores signals
- **Strategy Agent** — suggests trading strategies based on analyzed data

This modular separation allows the system to scale and adapt to new research needs.

## 📦 Installation
Clone the repository:
```bash
git clone https://github.com/gabriele-ts/MultiAgent-Quant-Research.git
cd MultiAgent-Quant-Research
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Usage
You can run the main agent from the project root:
```bash
python agent.py
```

Agents automatically coordinate to:
1. Ticker verification
2. Fetch or receive data
3. Clean and normalize it
4. Analyze and transform it
5. Produce insights and strategy suggestions
6. this is an example of the output it produce:
   Market Regime: High Volatility, Oscillatory with Downside Risk.

   Key Metrics Used:
   
   Short-term Kurtosis (0.2653) & Long-term Kurtosis (2.0818): Both are less than 3, indicating a normal-like return distribution, which generally supports trend-following strategies.
   Short-term Skewness (-0.5987): This is significantly negative, signaling a higher probability of large negative returns or "downside crashes," necessitating defensive measures.
   Volatility Regime: Short-Term Volatility (44.1792%) is greater than Long-Term Volatility (38.1004%), suggesting an extended move is in progress, which often precedes a period of consolidation or reversal, and warrants reduced position sizing.
   Short-term Autocorrelation:
   Lag 1 (0.158765): Positive and statistically significant (above 0.026157), indicating short-term momentum.
   Lag 2 (-0.092161), Lag 4 (-0.092385): Negative and statistically significant, suggesting mean-reversion tendencies after initial momentum.
   Lag 3 (0.050166), Lag 5 (0.031609): Positive and statistically significant, indicating an oscillatory pattern.
   Long-term Autocorrelation:
   Lag 1 (-0.02679): Negative and statistically significant, reinforcing mean-reversion tendencies over a slightly longer horizon.
   Proposed Strategy: Adaptive Momentum with Mean-Reversion Reversal and Volatility-Adjusted Sizing
   
   Theoretical Justification: The market exhibits a dual nature: initial short-term momentum (positive ACF at lag 1) followed by a tendency to revert (negative ACF at subsequent lags). This suggests an oscillatory regime where trends can form but are often met with reversals. The normal-like kurtosis supports trend-following, but the critical short-term negative skewness and elevated short-term volatility highlight significant downside risk and the need for cautious sizing. This strategy aims to capture the initial upward momentum while actively managing the risk of sharp reversals and protecting against potential crashes.
   
   Entry Logic (Long Only):
   
   Momentum Confirmation: Initiate a long position when a clear short-term upward momentum signal is observed. This could be a price breakout above a recent resistance level, a short-term moving average (e.g., 9-period EMA) crossing above a longer-term moving average (e.g., 21-period EMA), or a series of consecutive strong positive daily closes.
   Volatility Filter: Ensure that the current short-term annualized volatility (e.g., 20-day rolling volatility) is not in an extreme upper percentile (e.g., above the 80th percentile of its 60-day historical range). This helps avoid entering during periods of potentially unsustainable, highly extended moves that are prone to sharp corrections.
   Exit Logic:
   
   Mean-Reversion Reversal Signal: Exit the long position if the market shows signs of mean-reversion after an initial upward move. This can be triggered by:
   A significant bearish engulfing candlestick pattern or a "shooting star" after an uptrend.
   The short-term moving average (e.g., 9-period EMA) crossing back below the longer-term moving average (e.g., 21-period EMA).
   A predefined profit target is reached, taking advantage of the initial momentum before the market's mean-reverting tendencies dominate.
   Dynamic Stop-Loss: Implement a stop-loss order based on a multiple of the Average True Range (ATR) to adapt to current volatility. For example, place a stop-loss at 1.5 to 2 times the 14-period ATR below the entry price or a recent swing low. This protects against the negative skewness and potential sharp downside movements.
   Risk Management:
   
   Volatility-Adjusted Sizing: Dynamically adjust the position size inversely proportional to the current short-term annualized volatility. When short-term volatility is high (as indicated by Short-Term Vol > Long-Term Vol), reduce the capital allocated to the trade. For instance, if the current 20-day annualized volatility is 44.18% and the target volatility for a standard position is 38.10%, the position size should be scaled down by a factor of (38.10 / 44.18) to maintain a consistent risk exposure.
   Defensive Hedging: Given the significant short-term negative skewness, consider tactical hedging strategies. This could involve purchasing out-of-the-money put options on BTC or a highly correlated asset during periods of strong upward moves or when market uncertainty increases, to provide protection against potential "crash" events.
   Maximum Drawdown Limit: Define a strict maximum drawdown percentage for the overall portfolio or per trade, and adhere to it rigorously to prevent catastrophic losses from the identified downside risks.



## 🛠 Extending the System
You can easily add new agents:
1. Create a new file in `sub_agents/`
2. Define its tools and capabilities
3. Register it in `agent.py`


