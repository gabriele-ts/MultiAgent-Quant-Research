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

## 🛠 Extending the System
You can easily add new agents:
1. Create a new file in `sub_agents/`
2. Define its tools and capabilities
3. Register it in `agent.py`


