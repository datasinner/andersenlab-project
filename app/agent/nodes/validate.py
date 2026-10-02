"""validate_rule: deterministic checks on an extracted rule.

Schema problems were already caught by Pydantic during extraction (they
arrive here as feedback with no rule). This node adds grounding: every
number must appear in the excerpts the rule cites, every quote must be
verbatim, and every cited chunk must be part of the evidence.

Before checking, a citation whose quote is verbatim in a different evidence
chunk than the one it names is pointed at that chunk: the model quoted the
document correctly but mixed up neighbouring chunk ids. Quotes found in no
evidence chunk are left alone and fail the check.
"""

from app.agent.state import AgentStep, CompileState, Excerpt
from app.rules.dsl import ChargeRule, Citation
from app.rules.grounding import check_grounding, quote_in


def validate(state: CompileState) -> dict:
    charge = state["charge"]
    rule = state.get("rule")
    if rule is None:
        problems = list(state.get("feedback") or ["The previous answer could not be read."])
    else:
        rule = repair_citations(rule, state["evidence"])
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
    if rule is not None:
        update["rule"] = rule
        if not problems:
            update["grounded_rule"] = rule
    return update


def repair_citations(rule: ChargeRule, evidence: list[Excerpt]) -> ChargeRule:
    def repaired(citation: Citation | None) -> Citation | None:
        if citation is None:
            return None
        cited = next((e for e in evidence if e.chunk_id == citation.chunk_id), None)
        if cited is not None and quote_in(citation.quote, cited.text):
            return citation
        holder = next((e for e in evidence if quote_in(citation.quote, e.text)), None)
        if holder is None:
            return citation
        return citation.model_copy(
            update={
                "chunk_id": holder.chunk_id,
                "section_ref": holder.section_ref,
                "page": holder.page,
            }
        )

    return rule.model_copy(
        update={
            "citations": [repaired(c) for c in rule.citations],
            "exemptions": [
                e.model_copy(update={"citation": repaired(e.citation)}) for e in rule.exemptions
            ],
            "adjustments": [
                a.model_copy(update={"citation": repaired(a.citation)}) for a in rule.adjustments
            ],
        }
    )
