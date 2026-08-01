"""
Smoke tests for app.py's Flask routes that don't require a loaded model.
Endpoints that DO need MindBridge() (e.g. /chat) are covered by mocking
MindBridge entirely — CI should never try to download/run real DistilBERT.
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_status_endpoint_returns_200():
    client = _make_client()
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "retrain_running" in body


def test_retrain_status_endpoint_returns_200():
    client = _make_client()
    resp = client.get("/retrain_status")
    assert resp.status_code == 200


@patch("app.MindBridge")
def test_start_endpoint_creates_session(mock_mindbridge):
    """
    /start should work even with a fully mocked MindBridge instance —
    verifies the Flask routing/session logic independent of the ML model.
    """
    mock_bot = MagicMock()
    mock_bot.memory.is_returning.return_value = False
    mock_bot.memory.sessions.return_value = 1
    mock_bot.memory.total_messages.return_value = 0
    mock_bot.memory.data = {}
    mock_bot.memory.calibrator.to_dict.return_value = {}
    mock_mindbridge.return_value = mock_bot

    client = _make_client()
    resp = client.post(
        "/start",
        data=json.dumps({"name": "test_user", "memory": {}}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["name"] == "test_user"
    assert "greeting" in body
