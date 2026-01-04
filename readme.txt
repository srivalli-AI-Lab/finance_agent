# Finance Multi-Agent System

## Overview
This project is a web-based financial advisory application built with Flask and a multi-agent system powered by LangGraph and LangChain. It provides specialized agents for financial tasks:
- **Finance RAG Agent**: Answers general financial education queries using a knowledge base of curated articles.
- **Portfolio Agent**: Analyzes uploaded investment portfolios, calculating values, allocations, risks, and suggestions.
- **Market Analysis Agent**: Provides real-time insights on stocks using Alpha Vantage API, including metrics, trends, and recommendations.
- **Goal Planning Agent**: Helps with financial goal modeling, suggesting investment strategies based on user inputs like target amount and timeline.
- **Router**: Classifies queries and directs them to the appropriate agent.

The backend uses OpenAI for LLM tasks (e.g., classification, parsing), Alpha Vantage for market data, and FAISS for vector search in the RAG agent. The frontend is a simple HTML interface for chatting and uploading portfolios.

Architecture:
- **Frontend**: Flask serves an index.html with chat and upload features.
- **Backend**: LangGraph workflow routes queries; agents process and return markdown-formatted responses.
- **Data Flow**: User queries via /chat; portfolio uploads via /upload (stored in session); agents access DataFrame if uploaded.
- **Dependencies**: Relies on environment variables for API keys; caching and retries handle API limits.

## Setup Instructions
### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- API Keys:
  - OpenAI (for LLMs): Sign up at [platform.openai.com](https://platform.openai.com) and get an API key.
  - Alpha Vantage (for market data): Free tier at [alphavantage.co](https://www.alphavantage.co/support/#api-key).

### Installation
1. Clone the repository (or create the project structure if not in repo):
   ```
   git clone <repo-url>  # If applicable
   cd freddie_agent_flask
   ```

2. Create and activate a virtual environment:
   - Windows:
     ```
     python -m venv freddie_agent
     freddie_agent\Scripts\activate
     ```
   - macOS/Linux:
     ```
     python -m venv freddie_agent
     source freddie_agent/bin/activate
     ```

3. Install dependencies:
   ```
   pip install flask werkzeug pandas python-dotenv openai langchain-openai langchain-community langgraph faiss-cpu tenacity requests numpy-financial pydantic
   ```

4. Create a `.env` file in the root directory with your keys:
   ```
   OPENAI_API_KEY=sk-your-openai-key
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
   SECRET_KEY=your-random-secret-key  # Generate with: python -c "import os; print(os.urandom(24).hex())"
   USER_AGENT="FinanceAgent/1.0 (your.email@example.com)"  # Optional for DuckDuckGo
   ```

5. (Optional) Pre-build FAISS index for RAG agent to avoid startup delay:
   - Run: `python -c "from agents.rag_agent import build_kb; build_kb().save_local('faiss_index')"`

6. Run the app:
   ```
   python app.py
   ```
   - Access at http://localhost:5000.

## API Documentation
The app exposes a simple REST API via Flask.

### Routes
- **GET /**: Serves the main HTML interface (index.html) for chat and upload.

- **POST /upload**:
  - **Description**: Upload a portfolio Excel file (.xls or .xlsx) with columns 'Symbol' (stock ticker) and 'Quantity' (number of shares).
  - **Body**: Multipart form with 'file' field.
  - **Responses**:
    - 200: {"success": true, "filename": "uploaded_file.xlsx"}
    - 400: {"error": "Reason (e.g., missing columns, invalid file)"}
  - **Example (cURL)**:
    ```
    curl -X POST -F "file=@portfolio.xlsx" http://localhost:5000/upload
    ```

- **POST /chat**:
  - **Description**: Send a query to the agent system. If a portfolio is uploaded, it's available for portfolio-related queries.
  - **Body**: JSON {"message": "Your query here"}
  - **Responses**:
    - 200: {"response": "Agent's markdown-formatted answer"}
    - 400/500: {"error": "Reason"}
  - **Example (cURL)**:
    ```
    curl -X POST -H "Content-Type: application/json" -d '{"message": "Analyze MSFT"}' http://localhost:5000/chat
    ```

## Usage Examples
1. **Start the App**:
   - Run `python app.py` and open http://localhost:5000.
   - Upload a portfolio Excel (e.g., columns: Symbol | Quantity → AAPL | 10; MSFT | 5).

2. **Query Examples**:
   - General Finance: "What is budgeting?" → Routed to RAG agent, cites sources.
   - Portfolio: "Analyze my portfolio" → After upload, shows holdings, sectors, risk.
   - Market: "Can I buy MSFT?" → Extracts ticker, provides analysis with metrics/news.
   - Goal: "Reach 1 million in 10 years with $5k monthly" → Scenarios for conservative/moderate/aggressive strategies.

3. **Console Logs**:
   - Routing: "Query: 'can i buy msft' -> Routed to agent: MARKET_ANALYSIS"

## Troubleshooting Guide
- **Startup Slow**: FAISS index building (first run). Pre-build with script or reduce categories/max_results in rag_agent.py for testing.
- **API Key Errors**: Check .env; ensure keys are valid. For OpenAI: Billing enabled? For Alpha Vantage: Free tier limits (5 calls/min, 500/day).
- **Rate Limits (429/Too Many Requests)**: Alpha Vantage free tier is limited. Use premium key or add delays/retries (already in code). Cache helps reduce calls.
- **No Data/Invalid Ticker**: Check symbol (e.g., 'MSFT' not 'Microsoft'). APIs may fail on delisted stocks.
- **ModuleNotFoundError**: Run `pip install <missing-package>` in virtual env (e.g., faiss-cpu, tenacity).
- **Import Errors**: Ensure file structure: agents/ with router.py, rag_agent.py, etc.
- **CORS/Access Issues**: Run with `debug=True` for logs; browser console for JS errors.
- **General**: Delete faiss_index folder to rebuild KB; clear cache in agents if stale data.

For issues, check console logs or add prints. Project is for educational use—consult professionals for financial advice.