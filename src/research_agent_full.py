
"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.utils import get_today_str
from src.prompts import final_report_generation_prompt, untrusted_research_boundary
from src.state_scope import AgentState, AgentInputState
from src.research_agent_scope import clarify_with_user, write_research_brief
from src.multi_agent_supervisor import supervisor_agent
from src.model_config import model_for

# ===== Config =====

from langchain.chat_models import init_chat_model
writer_model = init_chat_model(model=model_for("writer"), max_tokens=32000)

# ===== FINAL REPORT GENERATION =====

from src.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    messages = [HumanMessage(content=final_report_prompt)]
    if os.getenv("EVAL_UNTRUSTED_BOUNDARY") == "1":
        messages.insert(0, SystemMessage(content=untrusted_research_boundary))
    final_report = await writer_model.ainvoke(messages)

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
