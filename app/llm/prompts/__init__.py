"""Prompt registry.

Every prompt lives in its own module in this package and is registered
here. Prompt text exists nowhere else in the codebase. Each prompt has a
version; bump it whenever the text changes in a way that could change the
model's output, because compiled rules are cached per prompt version.

Templates use string.Template ($name) placeholders rather than str.format,
so JSON examples with braces need no escaping.
"""

from dataclasses import dataclass
from string import Template

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, **values: str) -> list[BaseMessage]:
        """Fill both templates. A missing value raises KeyError rather than
        leaving a placeholder in the prompt."""
        return [
            SystemMessage(content=Template(self.system).substitute(values)),
            HumanMessage(content=Template(self.user).substitute(values)),
        ]


# Imported after PromptTemplate is defined: the prompt modules use it.
from app.llm.prompts import extract_rule  # noqa: E402

PROMPTS: dict[str, PromptTemplate] = {prompt.name: prompt for prompt in (extract_rule.PROMPT,)}


def versions(*names: str) -> str:
    """A stable identifier for a combination of prompt versions, used in
    cache keys: "critique@1+extract_rule@2"."""
    return "+".join(sorted(PROMPTS[name].id for name in names))
