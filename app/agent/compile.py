"""The rule-compilation graph for one charge at one port.

    research → extract_rule → validate_rule ─ok─→ critique ─approved─→ end
                    ↑               │ problems          │ revise
                    └───────────────┴───────────────────┘
                     (at most AGENT_MAX_REVISIONS re-extractions)

When revisions run out, the last rule that passed validation is kept and
flagged low_confidence (the critic's open issues are reported with it); if no
extraction ever passed validation, the charge fails.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes.critique import critique
from app.agent.nodes.extract import extract
from app.agent.nodes.research import research
from app.agent.nodes.validate import validate
from app.agent.state import AgentStep, CompileState
from app.llm.client import LLMClient


def build_compile_graph(
    llm: LLMClient, *, max_tool_calls: int, max_revisions: int
) -> CompiledStateGraph:
    async def research_node(state: CompileState) -> dict:
        return await research(state, llm=llm, max_tool_calls=max_tool_calls)

    async def extract_node(state: CompileState) -> dict:
        return await extract(state, llm=llm)

    async def critique_node(state: CompileState) -> dict:
        return await critique(state, llm=llm)

    def after_validation(state: CompileState) -> str:
        if not state.get("feedback"):
            return "critique"
        return "extract_rule" if state["revisions"] <= max_revisions else "stop_revising"

    def after_critique(state: CompileState) -> str:
        if state.get("outcome"):
            return END
        return "extract_rule" if state["revisions"] <= max_revisions else "stop_revising"

    graph = StateGraph(CompileState)
    graph.add_node("research", research_node)
    graph.add_node("extract_rule", extract_node)
    graph.add_node("validate_rule", validate)
    graph.add_node("critique", critique_node)
    graph.add_node("stop_revising", stop_revising)
    graph.add_edge(START, "research")
    graph.add_edge("research", "extract_rule")
    graph.add_edge("extract_rule", "validate_rule")
    graph.add_conditional_edges(
        "validate_rule",
        after_validation,
        {"critique": "critique", "extract_rule": "extract_rule", "stop_revising": "stop_revising"},
    )
    graph.add_conditional_edges(
        "critique",
        after_critique,
        {END: END, "extract_rule": "extract_rule", "stop_revising": "stop_revising"},
    )
    graph.add_edge("stop_revising", END)
    return graph.compile()


def stop_revising(state: CompileState) -> dict:
    grounded = state.get("grounded_rule")
    outcome = "low_confidence" if grounded is not None else "failed"
    step = AgentStep(
        node="stop_revising",
        charge_id=state["charge"].charge_id,
        output={"outcome": outcome, "open_issues": state.get("feedback", [])},
    )
    return {"outcome": outcome, "rule": grounded, "steps": [step]}
