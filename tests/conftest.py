"""
Pytest configuration for the trading-engine suite.

Ensures the tests directory is importable (so `import _synth` works regardless of
the invoking CWD) and exposes a couple of broadly useful fixtures. The heavy
lifting -- deterministic synthetic sessions and planted price paths -- lives in
_synth.py so each test states its adversarial intent, not its boilerplate.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def rng():
    """A seeded NumPy Generator. Determinism is non-negotiable for these tests."""
    import numpy as np

    return np.random.default_rng(0)
