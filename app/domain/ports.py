"""Matching a requested port name against the ports a tariff document covers.

Port names are written many ways ("Northhaven", "Port of Northhaven", "West
Quay / East Quay", "Southbay Harbour"), so names are compared as sets of
words, ignoring generic words such as "port" and "of":
1. an exact match on a name or alias wins;
2. otherwise a port whose words contain all the requested words, or whose
   words are all contained in the request, matches;
3. if several ports match equally, the request is ambiguous and nothing
   matches.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

_WORD = re.compile(r"[^\W_]+")
_GENERIC = {"port", "ports", "of", "the", "harbour", "harbor"}


def port_words(name: str) -> frozenset[str]:
    return frozenset(word for word in _WORD.findall(name.casefold()) if word not in _GENERIC)


def match_port(query: str, ports: Sequence[Mapping[str, Any]]) -> str | None:
    """The canonical name of the port `query` refers to, or None."""
    wanted = port_words(query)
    if not wanted:
        return None

    exact: list[str] = []
    partial: list[str] = []
    for port in ports:
        name = str(port["name"])
        variants = [port_words(name)] + [port_words(alias) for alias in port.get("aliases", [])]
        if any(variant == wanted for variant in variants):
            exact.append(name)
        elif any(variant and (wanted <= variant or variant <= wanted) for variant in variants):
            partial.append(name)

    for candidates in (exact, partial):
        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            return None
    return None
