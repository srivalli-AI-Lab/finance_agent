# agents/router.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from .rag_agent import finance_rag_node
from .portfolio_agent import portfolio_node
from .market_agent import market_node
from .goal_agent import goal_node

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    df: Optional[pd.DataFrame]
    next: str

def router_node(state: AgentState):
    query = state["messages"][-1].content
    prompt = f"""
    You are a router that classifies user queries into exactly one of the following categories based on the best match:

    - FINANCE_RAG: For general financial education queries, such as explanations of financial concepts, terms, strategies, or advice on personal finance topics like budgeting, investing basics, taxes, retirement, etc.
    - PORTFOLIO: For queries involving reviewing, analyzing, or providing advice on a user's investment portfolio, including asset allocation, risk assessment, performance evaluation, or rebalancing suggestions.
    - MARKET_ANALYSIS: For queries requesting real-time or current market insights, stock prices, trends, economic news, forecasts, or analysis of specific markets, sectors, or assets.
    - GOAL_PLANNING: For queries about setting, planning, or tracking financial goals, such as saving for a house, education, retirement planning, debt reduction strategies, or long-term financial modeling.

    Choose EXACTLY ONE category that best fits the query. Respond only with the category name (e.g., FINANCE_RAG, PORTFOLIO, MARKET_ANALYSIS, or GOAL_PLANNING). Do not add any explanations or additional text.

    Query: {query}
    """
    category = llm.invoke(prompt).content.strip().upper()
    # No need for mapping; assume LLM returns the exact category name
    selected = category if category in ["FINANCE_RAG", "PORTFOLIO", "MARKET_ANALYSIS", "GOAL_PLANNING"] else "FINANCE_RAG"
    print(f"Query: '{query}' -> Routed to agent: {selected}")  # Log the selected agent/category
    return {"next": selected}

def create_router():
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("finance_rag", finance_rag_node)
    workflow.add_node("portfolio", portfolio_node)
    workflow.add_node("market", market_node)
    workflow.add_node("goal", goal_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next"],
        {
            "FINANCE_RAG": "finance_rag",
            "PORTFOLIO": "portfolio",
            "MARKET_ANALYSIS": "market",
            "GOAL_PLANNING": "goal",
        }
    )
    for node in ["finance_rag", "portfolio", "market", "goal"]:
        workflow.add_edge(node, END)

    return workflow.compile()

router_graph = create_router()
def get_router_agent(): return router_graph