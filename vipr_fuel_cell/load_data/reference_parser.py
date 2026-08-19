"""Parser for the legacy INI-like acceptance reference files."""

from __future__ import annotations

import re
from pathlib import Path

from vipr_fuel_cell.constants import REFERENCE_NAME_MAPPING

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_]\w*(?:\.\w+)*)\s*=\s*(.*?)\s*;?\s*$")
_VECTOR = re.compile(rf"^\[\s*(?:{_NUMBER}\s+)*({_NUMBER})\s*\]$")
_NUMBER_ONLY = re.compile(rf"^({_NUMBER})$")
_REFERENCE = re.compile(r"^([A-Za-z_]\w*(?:\.\w+)*)$")
_ADDITION = re.compile(rf"^([A-Za-z_]\w*(?:\.\w+)*)\s*\+\s*({_NUMBER})$")


def parse_reference_file(path: Path) -> dict[str, float]:
    """Return mapped physical operating-parameter references from an INI file."""
    expressions: dict[str, tuple[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("_"):
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        key, expression = match.groups()
        expression = expression.rstrip(";").strip()

        if vector := _VECTOR.match(expression):
            expressions[key] = ("value", float(vector.group(1)))
        elif number := _NUMBER_ONLY.match(expression):
            expressions[key] = ("value", float(number.group(1)))
        elif addition := _ADDITION.match(expression):
            expressions[key] = (
                "addition",
                (addition.group(1), float(addition.group(2))),
            )
        elif reference := _REFERENCE.match(expression):
            expressions[key] = ("reference", reference.group(1))

    resolved: dict[str, float] = {
        key: float(payload)
        for key, (kind, payload) in expressions.items()
        if kind == "value"
    }
    pending = dict(expressions)
    for _ in range(len(pending) + 1):
        changed = False
        for key, (kind, payload) in pending.items():
            if key in resolved:
                continue
            if kind == "reference" and payload in resolved:
                resolved[key] = resolved[str(payload)]
                changed = True
            elif kind == "addition":
                reference, increment = payload
                if reference in resolved:
                    resolved[key] = resolved[reference] + float(increment)
                    changed = True
        if not changed:
            break

    return {
        mapped_name: resolved[source_name]
        for source_name, mapped_name in REFERENCE_NAME_MAPPING.items()
        if source_name in resolved
    }
