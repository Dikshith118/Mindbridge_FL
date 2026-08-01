"""
tests/conftest.py
==================
Shared pytest fixtures.

The single most important thing this file does: it prevents CI from trying
to download/load the real DistilBERT model (66M params) on every run. That
would make CI slow, flaky on network hiccups, and pointless — CI should
verify code correctness, not model weights.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chatbot"))

os.environ.setdefault("MINDBRIDGE_TEST_MODE", "1")


@pytest.fixture
def normalize_text():
    from chatbot import normalize_text as _fn
    return _fn


@pytest.fixture
def compute_risk_score():
    from chatbot import compute_risk_score as _fn
    return _fn


@pytest.fixture
def is_crisis():
    from chatbot import is_crisis as _fn
    return _fn


@pytest.fixture
def extract_topic():
    from chatbot import extract_topic as _fn
    return _fn
