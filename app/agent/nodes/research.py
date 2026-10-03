"""research: a bounded tool-calling loop that gathers the evidence for one charge.

The catalogue's sections for the charge are opened before the model's first
turn (they always matter, and it saves a round trip). The model then
searches, reads and looks up whatever else it needs, and ends by submitting
the chunk ids it relied on. If it stops without submitting, or runs out of
tool calls, everything it saw becomes the evidence.
"""

import time

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import ValidationError

from app.agent.state import AgentStep, CompileState, Excerpt
from app.agent.tools import RESEARCH_TOOLS, SubmitEvidence, ToolExecutor
from app.llm.client import LLMClient
from app.llm.prompts import PROMPTS


async def research(state: CompileState, *, llm: LLMClient, max_tool_calls: int) -> dict:
    charge = state["charge"]
    tools: ToolExecutor = state["tools"]
    steps: list[AgentStep] = []

    opened = []
    for ref in charge.section_refs:
        opened.append(await tools.read_section(ref))
        steps.append(
            AgentStep(
                node="research", charge_id=charge.charge_id, tool="ReadSection", input_summary=ref
            )
        )
    preopened_ids = set(tools.seen)

    messages: list[BaseMessage] = PROMPTS["research"].render(
        port=state["port"],
        charge_name=charge.name,
        description=charge.description,
        section_refs=", ".join(charge.section_refs),
        opening_excerpts="\n\n".join(opened),
    )

    submitted: SubmitEvidence | None = None
    tool_calls = 0
    while submitted is None and tool_calls < max_tool_calls:
        started = time.monotonic()
        turn = await llm.generate_with_tools(messages, RESEARCH_TOOLS, name="research")
        messages.append(turn.message)
        steps.append(
            AgentStep(
                node="research",
                charge_id=charge.charge_id,
                input_summary=f"turn with {len(turn.message.tool_calls)} tool call(s)",
                latency_ms=int((time.monotonic() - started) * 1000),
                prompt_tokens=turn.usage.prompt_tokens,
                completion_tokens=turn.usage.completion_tokens,
            )
        )
        if not turn.message.tool_calls:
            break
        for call in turn.message.tool_calls:
            if call["name"] == SubmitEvidence.__name__:
                try:
                    submitted = SubmitEvidence.model_validate(call["args"])
                    result = "Evidence received."
                except ValidationError as exc:
                    result = f"Invalid evidence: {exc.errors()[0]['msg']}"
            else:
                tool_calls += 1
                result = await tools.execute(call["name"], call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
            steps.append(
                AgentStep(
                    node="research",
                    charge_id=charge.charge_id,
                    tool=call["name"],
                    input_summary=_summarize(call["args"]),
                )
            )

    if submitted is not None:
        chosen = preopened_ids | {
            chunk_id for chunk_id in submitted.chunk_ids if chunk_id in tools.seen
        }
        notes = submitted.notes
    else:
        chosen = set(tools.seen)
        notes = "(research ended without submitting evidence; all excerpts seen are included)"
    evidence: list[Excerpt] = [tools.seen[chunk_id] for chunk_id in sorted(chosen)]
    return {"evidence": evidence, "research_notes": notes, "steps": steps}


def _summarize(arguments: dict) -> str:
    text = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return text[:300]
