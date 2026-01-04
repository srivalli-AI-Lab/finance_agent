# agents/goal_agent.py
import numpy_financial as npf
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GoalInput(BaseModel):
    goal: float = Field(..., description="Target amount")
    years: float = Field(..., description="Years")
    monthly: float = Field(None, description="Monthly saving amount")
    return_rate: float = Field(7.0, description="Annual return % (default moderate)")

def goal_node(state):
    query = state["messages"][-1].content

    try:
        parser = llm.with_structured_output(GoalInput)
        parsed = parser.invoke(f"Extract goal, years, monthly saving (if mentioned), and return rate (if specified) from: {query}. Use defaults where not provided.")
    except Exception as e:
        return {"messages": [AIMessage(content=f"Parsing error: {str(e)}. Please provide goal, years, and optionally monthly savings or return rate.")]}

    goal = parsed.goal
    years = parsed.years
    monthly = parsed.monthly
    default_rate = parsed.return_rate

    scenarios = {
        "Conservative (Savings/Bonds)": {"rate": 4.0, "examples": "High-yield savings, BND (Bond ETF)"},
        "Moderate (Balanced Funds/ETFs)": {"rate": 7.0, "examples": "Target-date funds, VTI (Total Stock ETF) + BND mix"},
        "Aggressive (Stocks)": {"rate": 10.0, "examples": "Individual stocks, VOO (S&P 500 ETF), growth funds"}
    }

    response_parts = [f"**Goal: ${goal:,.0f} in {years} years**"]

    if monthly is not None:
        # Calculate future value for each scenario
        response_parts.append("\n**With monthly savings of ${monthly:,.0f}:**")
        for name, info in scenarios.items():
            r = info["rate"] / 100 / 12
            n = years * 12
            fv = -npf.fv(r, n, monthly, 0)
            reaches = fv >= goal
            shortfall = goal - fv if not reaches else 0
            extra_needed = abs(npf.pmt(r, n, 0, goal)) - monthly if not reaches else 0
            status = "Reaches goal" if reaches else f"Short by ${shortfall:,.0f} (increase monthly by ${extra_needed:,.0f})"
            response_parts.append(f"- **{name} ({info['rate']}% return)**: Future value ${fv:,.0f}. {status}. Suggestions: {info['examples']}.")
    else:
        # Calculate required monthly for each scenario
        response_parts.append("\n**Required monthly investments to reach goal:**")
        for name, info in scenarios.items():
            r = info["rate"] / 100 / 12
            n = years * 12
            required = abs(npf.pmt(r, n, 0, goal))
            response_parts.append(f"- **{name} ({info['rate']}% return)**: ${required:,.0f}/month. Suggestions: {info['examples']}.")

    response_parts.append("\n**Tips**: Diversify investments, consider risk tolerance, and consult a financial advisor. Start auto-investing today!")

    response = "\n".join(response_parts)
    return {"messages": [AIMessage(content=response)]}