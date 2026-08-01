"""
Unit tests for MindBridge's rule-based logic layers.
These are pure functions with no model dependency — fast, deterministic,
safe to run in CI on every commit.
"""


def test_normalize_fixes_common_typos(normalize_text):
    assert normalize_text("i feel sadd today") == "i feel sad today"
    assert "suicide" in normalize_text("i think about sucide sometimes")


def test_normalize_preserves_correct_text(normalize_text):
    assert normalize_text("i am happy") == "i am happy"


def test_crisis_detects_direct_phrases(is_crisis):
    assert is_crisis("i want to kill myself", "i want to kill myself") is True
    assert is_crisis("suicide is on my mind", "suicide is on my mind") is True


def test_crisis_ignores_unrelated_text(is_crisis):
    assert is_crisis("i had a great day", "i had a great day") is False


def test_crisis_catches_fuzzy_misspellings(is_crisis):
    # FUZZY_CRISIS_PATTERNS should catch common misspellings/obfuscations
    assert is_crisis("i want to kil myself", "i want to kil myself") is True


def test_risk_score_high_for_high_risk_phrase(compute_risk_score):
    score = compute_risk_score(
        "i feel worthless and want to disappear",
        emotion="sadness", confidence=0.8, history=[],
    )
    assert score >= 0.5


def test_risk_score_low_for_neutral_text(compute_risk_score):
    score = compute_risk_score(
        "the weather is nice today",
        emotion="neutral", confidence=0.5, history=[],
    )
    assert score < 0.3


def test_risk_score_rises_with_negative_history(compute_risk_score):
    no_history = compute_risk_score("i feel sad", "sadness", 0.7, [])
    with_history = compute_risk_score(
        "i feel sad", "sadness", 0.7,
        history=["sadness", "grief", "sadness", "fear", "sadness"],
    )
    assert with_history >= no_history


def test_extract_topic_detects_work(extract_topic):
    assert extract_topic("my boss gave me a deadline at work") == "work"


def test_extract_topic_detects_family(extract_topic):
    assert extract_topic("my mom and dad are visiting") == "family"


def test_extract_topic_returns_none_for_generic_text(extract_topic):
    assert extract_topic("i went to the store") is None
