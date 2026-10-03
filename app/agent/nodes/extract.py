"""extract_rule: evidence → ChargeRule (Structured Outputs).

On a revision the prompt also carries the previous answer and the problems
the validator or the critic found, so the model fixes rather than starts
over. Identity fields (charge id, port, currency) are set from what is
already known instead of trusting the model to copy them.
"""

import time

from app.agent.state import AgentStep, CompileState, render_excerpts
from app.llm.client import LLMClient, LLMOutputError
from app.llm.prompts import PROMPTS
from app.llm.prompts.extract_rule import quantity_glossary
from app.rules.dsl import ChargeRule


async def extract(state: CompileState, *, llm: LLMClient) -> dict:
    charge = state["charge"]
    messages = PROMPTS["extract_rule"].render(
        quantities=quantity_glossary(),
        port=state["port"],
        charge_name=charge.name,
        currency=state["currency"],
        research_notes=state.get("research_notes") or "(none)",
        excerpts=render_excerpts(state["evidence"]),
        feedback=feedback_block(state.get("rule"), state.get("feedback", [])),
    )
    started = time.monotonic()
    try:
        result = await llm.generate_structured(ChargeRule, messages, name="extract_rule")
    except LLMOutputError as exc:
        step = AgentStep(
            node="extract_rule",
            charge_id=charge.charge_id,
            output={"error": str(exc), "errors": exc.errors},
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return {"rule": None, "feedback": exc.errors or [str(exc)], "steps": [step]}

    rule = result.value.model_copy(
        update={
            "charge_id": charge.charge_id,
            "port_key": state["port"],
            "currency": state["currency"],
        }
    )
    step = AgentStep(
        node="extract_rule",
        charge_id=charge.charge_id,
        output={"components": len(rule.components), "status": rule.status},
        latency_ms=result.latency_ms,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )
    return {"rule": rule, "feedback": [], "steps": [step]}


def feedback_block(previous: ChargeRule | None, problems: list[str]) -> str:
    if not problems:
        return ""
    lines = ["", "Your previous answer had these problems. Fix them and answer again:"]
    lines.extend(f"- {problem}" for problem in problems)
    if previous is not None:
        lines.extend(["", "Previous answer:", previous.model_dump_json(indent=1)])
    return "\n".join(lines)
