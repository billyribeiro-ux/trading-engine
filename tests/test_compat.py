"""
Python 3.9 compatibility invariants.

The engine MUST run on the operator's Mac Python 3.9.6. These are meta-tests over
the source tree itself, locking the constraints that a 3.10+ idiom would silently
violate:

  * `slots=True` is forbidden (the brief's explicit grep invariant: must be 0).
  * No `match` statements (a SyntaxError on 3.9).
  * Every module imports cleanly on the running interpreter (catches evaluated
    `X | None` unions and other 3.10-isms that only blow up at import time).
  * Any file using `X | None`-style union hints carries
    `from __future__ import annotations`, so the union is a string, not evaluated.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import engine

ENGINE_DIR = Path(engine.__file__).resolve().parent
PY_FILES = sorted(ENGINE_DIR.rglob("*.py"))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_engine_tree_nonempty():
    # Guard the guard: if globbing finds nothing, the other tests are vacuous.
    assert len(PY_FILES) >= 20, f"only found {len(PY_FILES)} engine .py files"


def test_no_slots_true():
    """The locked invariant from the brief: grep slots=True must be 0."""
    offenders = [str(p) for p in PY_FILES if "slots=True" in _read(p)]
    assert offenders == [], f"slots=True found in: {offenders}"


def test_no_match_statements():
    """`match`/`case` soft-keyword statements are 3.10+ only."""
    offenders = []
    for p in PY_FILES:
        for node in ast.walk(ast.parse(_read(p), filename=str(p))):
            # ast.Match exists on 3.9's ast module as a node type only when parsed
            # on 3.10+; on 3.9 a match statement is a SyntaxError before we get
            # here. We still scan defensively for the class by name.
            if type(node).__name__ in {"Match", "match_case"}:
                offenders.append(str(p))
                break
    assert offenders == [], f"match statements in: {offenders}"


@pytest.mark.parametrize(
    "modname",
    [m.name for m in pkgutil.walk_packages(engine.__path__, "engine.")],
)
def test_module_imports(modname):
    """Every submodule imports on this interpreter (3.9)."""
    importlib.import_module(modname)


# A union hint with a builtin/None on either side -- overwhelmingly a type union,
# not a bitwise-or, so this won't false-positive on the current tree.
_UNION = re.compile(
    r"\b\w[\w\]]*\s*\|\s*(None|bool|int|float|str|bytes|list|dict|tuple|set)\b"
    r"|\bNone\s*\|\s*\w"
)


def test_union_hints_have_future_annotations():
    """Files using `X | None` hints must defer annotation evaluation on 3.9."""
    offenders = []
    for p in PY_FILES:
        src = _read(p)
        if _UNION.search(src) and "from __future__ import annotations" not in src:
            offenders.append(str(p))
    assert offenders == [], f"union hints without future-annotations: {offenders}"
