from fastapi.testclient import TestClient

from action_grounding_service.app.main import app


def test_config_update_round_trip(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
action_grounding_stage:
  grounding_url: "http://localhost:8000"
dotenv_path: ".env"
action_grounding_service: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ACTION_GROUNDING_CONFIG", str(config_path))

    async def fake_check(_config):
        return {
            "ocr": {"ok": True, "message": "ok"},
            "vlm": {"ok": True, "message": "ok"},
            "omniparser": {"ok": True, "message": "ok"},
        }

    monkeypatch.setattr("action_grounding_service.app.main.check_service_config", fake_check)

    with TestClient(app) as client:
        config = client.get("/config").json()
        config["ocr"]["model"] = "openai/gpt-5.4"
        config["ocr"]["env"] = {"OPENAI_API_KEY": "abc"}
        config["vlm"]["model"] = "openai/gpt-5.4-mini"
        config["omniparser"]["imgsz"] = 512
        config["omniparser"]["max_concurrent_requests"] = 2

        response = client.put("/config", json=config)
        check_response = client.post("/config/check", json=config)

    assert response.status_code == 200
    assert check_response.status_code == 200
    assert check_response.json()["ocr"]["ok"] is True
    saved = config_path.read_text(encoding="utf-8")
    assert "action_grounding_stage:" in saved
    assert 'dotenv_path: ".env"' in saved or "dotenv_path: .env" in saved
    assert "action_grounding_service:" in saved
    assert "model: openai/gpt-5.4" in saved
    assert "model: openai/gpt-5.4-mini" in saved
    assert "OPENAI_API_KEY: abc" in saved
    assert "max_concurrent_requests: 2" in saved
    assert "imgsz: 512" in saved
