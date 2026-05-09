from pathlib import Path

from fastapi.testclient import TestClient

import tts_service


client = TestClient(tts_service.app)


def test_health_returns_selected_voices():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["voices"]["miss_vivian"] == "af_bella"
    assert response.json()["voices"]["miss_ciel"] == "af_heart"
    assert response.json()["voices"]["miss_estelle"] == "bf_isabella"


def test_voices_returns_agent_profiles():
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json()["agents"]["miss_ciel"]["voice"] == "af_heart"
    assert response.json()["agents"]["miss_estelle"]["speed"] == 0.95


def test_synthesize_rejects_unknown_agent():
    response = client.post("/synthesize", json={"agent": "unknown", "text": "Hello"})

    assert response.status_code == 422


def test_synthesize_rejects_empty_and_long_text():
    empty = client.post("/synthesize", json={"agent": "miss_ciel", "text": "   "})
    long = client.post("/synthesize", json={"agent": "miss_ciel", "text": "x" * 301})

    assert empty.status_code == 422
    assert long.status_code == 422


def test_synthesize_sanitizes_text_and_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0, "text": None}
    monkeypatch.setattr(tts_service, "CACHE_DIR", tmp_path)

    def fake_generate(text: str, voice: str, speed: float, output_path: Path):
        calls["count"] += 1
        calls["text"] = text
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFfake-wave")

    monkeypatch.setattr(tts_service, "generate_audio", fake_generate)

    payload = {"agent": "miss_ciel", "text": "<b>Good try!</b>", "cache": True}
    first = client.post("/synthesize", json=payload)
    second = client.post("/synthesize", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["content-type"].startswith("audio/wav")
    assert calls["count"] == 1
    assert calls["text"] == "Good try!"
    assert "generated_audio" not in first.text


def test_cache_key_is_stable():
    assert tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi") == tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi")
