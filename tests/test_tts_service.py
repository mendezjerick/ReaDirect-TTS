import logging
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
    assert response.json()["humanization"]["text_humanizer_enabled"] is True


def test_voices_returns_agent_profiles():
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json()["agents"]["miss_ciel"]["voice"] == "af_heart"
    assert response.json()["agents"]["miss_ciel"]["speed"] == 0.94
    assert response.json()["agents"]["miss_ciel"]["speed_range"] == {"min": 0.92, "max": 0.96}
    assert response.json()["agents"]["miss_estelle"]["speed"] == 0.93


def test_ciel_voice_cannot_be_overridden_and_speed_is_clamped():
    prepared = tts_service.audio_path_for(
        tts_service.SynthesizeRequest(
            agent="miss_ciel",
            text="Good job.",
            voice="af_bella",
            speed=1.2,
        )
    )

    assert prepared.voice == "af_heart"
    assert prepared.speed == 0.96


def test_synthesize_rejects_unknown_agent():
    response = client.post("/synthesize", json={"agent": "unknown", "text": "Hello"})

    assert response.status_code == 422


def test_synthesize_rejects_empty_and_long_text():
    empty = client.post("/synthesize", json={"agent": "miss_ciel", "text": "   "})
    long = client.post("/synthesize", json={"agent": "miss_ciel", "text": "x" * 301})

    assert empty.status_code == 422
    assert long.status_code == 422


def test_synthesize_humanizes_sanitized_text_and_uses_cache(monkeypatch, tmp_path):
    calls = {"count": 0, "text": None}
    monkeypatch.setattr(tts_service, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")

    def fake_generate(text: str, voice: str, speed: float, output_path: Path, *args, **kwargs):
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
    assert calls["text"] == "Good try. Let's keep practicing together."
    assert "generated_audio" not in first.text


def test_cache_key_is_stable():
    assert tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi") == tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi")


def test_text_humanizer_expands_robotic_lines(monkeypatch):
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")
    profile = tts_service.AGENT_PROFILES["miss_ciel"]

    result = tts_service.prepare_tts_text(
        "miss_ciel",
        "Try again.",
        profile,
        tts_service.SynthesizeRequest(agent="miss_ciel", text="Try again."),
    )

    assert result[0] == "That's okay. Let's try that one more time."
    assert result[2].applied is True


def test_text_humanizer_preserves_protected_letters_words_and_choices():
    profile = tts_service.AGENT_PROFILES["miss_ciel"]

    for text in ["A", "cat", "A. cat B. dog"]:
        humanized, synthesis_text, text_result, delivery_result = tts_service.prepare_tts_text(
            "miss_ciel",
            text,
            profile,
            tts_service.SynthesizeRequest(agent="miss_ciel", text=text),
        )

        assert humanized == text
        assert synthesis_text == text
        assert text_result.applied is False
        assert delivery_result.applied is False


def test_text_humanizer_can_be_disabled():
    profile = tts_service.AGENT_PROFILES["miss_ciel"]
    request = tts_service.SynthesizeRequest(agent="miss_ciel", text="Good job.", humanize=False)

    humanized, synthesis_text, text_result, _ = tts_service.prepare_tts_text("miss_ciel", "Good job.", profile, request)

    assert humanized == "Good job."
    assert synthesis_text == "Good job."
    assert text_result.applied is False


def test_delivery_control_can_be_disabled():
    profile = tts_service.AGENT_PROFILES["miss_vivian"]
    request = tts_service.SynthesizeRequest(agent="miss_vivian", text="When you're ready say it", humanize=False, delivery_control=False)

    _, synthesis_text, _, delivery_result = tts_service.prepare_tts_text("miss_vivian", "When you're ready say it", profile, request)

    assert synthesis_text == "When you're ready say it"
    assert delivery_result.applied is False


def test_audio_post_processing_and_breaths_are_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_HUMANIZER_ENABLED", "false")
    prepared = tts_service.audio_path_for(
        tts_service.SynthesizeRequest(
            agent="miss_estelle",
            text="You did well today. Let's look at your result together.",
            audio_humanizer=False,
        )
    )

    assert prepared.voice == "bf_isabella"
    assert prepared.speed == 0.93
    assert tts_service.audio_humanizer_enabled(False) is False


def test_debug_logging_includes_original_and_humanized_text(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tts_service, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("TTS_DEBUG_LOGGING", "true")
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")
    caplog.set_level(logging.INFO, logger=tts_service.SERVICE_NAME)

    def fake_generate(text: str, voice: str, speed: float, output_path: Path, *args, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFfake-wave")

    monkeypatch.setattr(tts_service, "generate_audio", fake_generate)

    response = client.post("/synthesize", json={"agent": "miss_ciel", "text": "Good job.", "cache": False})

    assert response.status_code == 200
    assert '"original_text": "Good job."' in caplog.text
    assert '"humanized_text": "Good job. You said that clearly."' in caplog.text
