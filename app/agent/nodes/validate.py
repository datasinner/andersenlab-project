"""validate_rule: deterministic checks on an extracted rule.

Schema problems were already caught by Pydantic during extraction (they
arrive here as feedback with no rule). This node adds grounding: every
number must appear in the excerpts the rule cites, every quote must be
verbatim, and every cited chunk must be part of the evidence.
"""

from app.agent.state import AgentStep, CompileState
from app.rules.grounding import check_grounding


def validate(state: CompileState) -> dict:
    charge = state["charge"]
    rule = state.get("rule")
    if rule is None:
        problems = list(state.get("feedback") or ["The previous answer could not be read."])
    else:
        chunk_texts = {excerpt.chunk_id: excerpt.text for excerpt in state["evidence"]}
        problems = [
            f"{issue.path}: {issue.message}" for issue in check_grounding(rule, chunk_texts)
        ]

    revisions = state.get("revisions", 0) + (1 if problems else 0)
    step = AgentStep(
        node="validate_rule",
        charge_id=charge.charge_id,
        output={"problems": problems} if problems else {"grounded": True},
    )
    update: dict = {"feedback": problems, "revisions": revisions, "steps": [step]}
    if rule is not None and not problems:
        update["grounded_rule"] = rule
    return update
