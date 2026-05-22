"""The /widget.js loader is served with the right content type and embed logic."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_widget_js_served_as_javascript() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/widget.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    body = response.text
    assert "data-widget-id" in body
    assert "/config" in body  # fetches public config
    assert "iframe" in body  # injects an iframe
    assert "resize" in body  # postMessage resize handling
