"""Types the agent graph passes between nodes."""

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel

from app.models import ChargeCatalogueEntry
from app.rules.dsl import ChargeRule


@dataclass(frozen=True)
class Excerpt:
    """One chunk of the tariff document as the model sees it."""

    chunk_id: int
    section_ref: str
    page: int
    printed_page: str | None
    text: str

    def render(self) -> str:
        printed = f' printed_page="{self.printed_page}"' if self.printed_page else ""
        return (
            f'<tariff_excerpt chunk_id="{self.chunk_id}" section="{self.section_ref}" '
            f'page="{self.page}"{printed}>\n{self.text}\n</tariff_excerpt>'
        )

    def render_reference(self) -> str:
        """A pointer to an excerpt already shown in the conversation."""
        return (
            f'<tariff_excerpt chunk_id="{self.chunk_id}" section="{self.section_ref}" '
            f'page="{self.page}">(shown above)</tariff_excerpt>'
        )


def render_excerpts(excerpts: list[Excerpt]) -> str:
    return "\n\n".join(excerpt.render() for excerpt in excerpts) or "(none)"


@dataclass(frozen=True)
class ChargeSpec:
    """A charge to compile, as the catalogue describes it."""

    charge_id: str
    name: str
    description: str
    section_refs: list[str]
    payer: str
    trigger: str

    @classmethod
    def from_catalogue(cls, entry: ChargeCatalogueEntry) -> "ChargeSpec":
        return cls(
            charge_id=entry.charge_id,
            name=entry.name,
            description=entry.description,
            section_refs=list(entry.section_refs),
            payer=entry.payer,
            trigger=entry.trigger,
        )


@dataclass(frozen=True)
class AgentStep:
    """One thing the agent did, for the audit trail (agent_step rows)."""

    node: str
    charge_id: str | None = None
    tool: str | None = None
    input_summary: str | None = None
    output: dict[str, Any] | None = None
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ReviewIssue(BaseModel):
    severity: Literal["blocking", "minor"]
    problem: str


class Critique(BaseModel):
    issues: list[ReviewIssue]

    @property
    def blocking(self) -> list[str]:
        return [issue.problem for issue in self.issues if issue.severity == "blocking"]

    @property
    def minor(self) -> list[str]:
        return [issue.problem for issue in self.issues if issue.severity == "minor"]


CompileOutcomeStatus = Literal["approved", "low_confidence", "failed"]


class CompileState(TypedDict, total=False):
    # Inputs
    charge: ChargeSpec
    port: str
    currency: str
    tools: Any  # app.agent.tools.ToolExecutor (kept untyped to avoid an import cycle)
    # Working state
    evidence: list[Excerpt]
    research_notes: str
    rule: ChargeRule | None
    grounded_rule: ChargeRule | None  # the latest rule that passed validation
    best_rule: ChargeRule | None  # the reviewed rule with the fewest blocking issues
    best_issues: list[str]  # that rule's blocking issues
    feedback: list[str]
    revisions: int
    critique: Critique | None
    review_notes: list[str]  # minor issues from the last review
    # Result
    outcome: CompileOutcomeStatus
    steps: Annotated[list[AgentStep], operator.add]
