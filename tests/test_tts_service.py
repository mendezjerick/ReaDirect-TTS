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
    assert response.json()["pipeline"]["warmup_on_startup"] is False
    assert "loaded" in response.json()["pipeline"]
    assert response.json()["engines"]["default"] == "kokoro"
    assert response.json()["engines"]["index_tts2"]["enabled"] is False
    assert response.json()["engines"]["index_tts2"]["fallback_to_kokoro"] is True
    assert response.json()["references"]["manifest_path"].endswith("storage\\tts\\references\\manifest.json") or response.json()["references"]["manifest_path"].endswith("storage/tts/references/manifest.json")
    assert response.json()["humanization"]["auto_prompt_extension_enabled"] is False
    assert response.json()["humanization"]["curated_prompts_enabled"] is True
    assert response.json()["humanization"]["text_humanizer_enabled"] is False


def test_voices_returns_agent_profiles():
    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json()["agents"]["miss_ciel"]["voice"] == "af_heart"
    assert response.json()["agents"]["miss_ciel"]["agent_key"] == "ciel"
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
    assert calls["text"] == "That's okay, let's try that one more time. This one can be tricky, but we can slow it down together."
    assert "generated_audio" not in first.text


def test_cache_bypass_forces_regeneration(monkeypatch, tmp_path):
    calls = {"count": 0}
    monkeypatch.setattr(tts_service, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("TTS_CACHE_BYPASS", "true")
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")

    def fake_generate(text: str, voice: str, speed: float, output_path: Path, *args, **kwargs):
        calls["count"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFfake-wave")

    monkeypatch.setattr(tts_service, "generate_audio", fake_generate)

    payload = {"agent": "miss_vivian", "text": "Listen carefully.", "cache": True}
    first = client.post("/synthesize", json=payload)
    second = client.post("/synthesize", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 2


def test_cache_key_is_stable():
    assert tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi") == tts_service.cache_key("miss_ciel", "af_heart", 1.0, "Hi")


def test_curated_prompts_replace_known_short_lines(monkeypatch):
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")
    profile = tts_service.AGENT_PROFILES["miss_ciel"]

    result = tts_service.prepare_tts_text(
        "miss_ciel",
        "Try again.",
        profile,
        tts_service.SynthesizeRequest(agent="miss_ciel", text="Try again."),
    )

    assert result[0] == "That's okay, let's try that one more time. This one can be tricky, but we can slow it down together."
    assert result[2].applied is True
    assert result[2].reason.startswith("curated:")


def test_auto_prompt_extension_requires_flag(monkeypatch):
    monkeypatch.setenv("TTS_CURATED_PROMPTS_ENABLED", "false")
    monkeypatch.setenv("TTS_AUTO_PROMPT_EXTENSION_ENABLED", "true")
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")
    profile = tts_service.AGENT_PROFILES["miss_ciel"]

    result = tts_service.prepare_tts_text(
        "miss_ciel",
        "Try again.",
        profile,
        tts_service.SynthesizeRequest(agent="miss_ciel", text="Try again."),
    )

    assert result[0] == "That's okay, let's try that one more time."
    assert result[2].applied is True


def test_prosody_intent_classifier_uses_humanized_text(monkeypatch):
    monkeypatch.setenv("TTS_TEXT_HUMANIZER_VARIATION_ENABLED", "false")
    prepared = tts_service.audio_path_for(
        tts_service.SynthesizeRequest(agent="miss_ciel", text="Good job.")
    )

    assert prepared.intent_result.intent == "happy_praise"
    assert prepared.intent_result.agent_key == "ciel"


def test_ciel_intro_intent_is_supported(monkeypatch):
    monkeypatch.setenv("TTS_EMOTION_PROMPT_INTRO", "warm, friendly, welcoming, gentle")
    prepared = tts_service.audio_path_for(
        tts_service.SynthesizeRequest(
            agent="miss_ciel",
            text="Hi, I'm Miss Ciel. I'll read with you today.",
            metadata={"prosody_intent": "intro"},
        )
    )

    assert prepared.intent_result.intent == "intro"
    assert prepared.emotion_prompt == "warm, friendly, welcoming, gentle"
    assert prepared.voice == "af_heart"


def test_reference_selection_is_deterministic_with_multiple_ciel_files(tmp_path):
    reference_root = tmp_path / "references"
    folder = reference_root / "ciel" / "friendly_encouragement"
    folder.mkdir(parents=True)
    for index in range(1, 4):
        (folder / f"ciel_friendly_encouragement_{index:02d}.wav").write_bytes(b"RIFFfake-wave")

    manifest = tts_service.ReferenceManifest(
        manifest_path=reference_root / "manifest.json",
        audio_root=reference_root,
        data={
            "ciel": {
                "friendly_encouragement": [
                    f"ciel/friendly_encouragement/ciel_friendly_encouragement_{index:02d}.wav"
                    for index in range(1, 4)
                ]
            }
        },
        loaded=True,
    )

    first = manifest.select("ciel", "friendly_encouragement", "same text")
    second = manifest.select("ciel", "friendly_encouragement", "same text")

    assert first.path == second.path
    assert first.relative_path in {
        "ciel/friendly_encouragement/ciel_friendly_encouragement_01.wav",
        "ciel/friendly_encouragement/ciel_friendly_encouragement_02.wav",
        "ciel/friendly_encouragement/ciel_friendly_encouragement_03.wav",
    }


def test_missing_reference_manifest_is_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(tts_service.tts_config, "reference_manifest_path", lambda: tmp_path / "missing_manifest.json")
    monkeypatch.setattr(tts_service.tts_config, "reference_audio_root", lambda: tmp_path / "references")

    prepared = tts_service.audio_path_for(
        tts_service.SynthesizeRequest(agent="miss_estelle", text="Let's look at your result together.")
    )

    assert prepared.reference_selection.manifest_loaded is False
    assert prepared.reference_selection.fallback_reason == "manifest_missing"
    assert prepared.engine == "kokoro"


def test_expressive_request_falls_back_to_kokoro_when_adapter_missing(monkeypatch, tmp_path):
    calls = {"count": 0}
    monkeypatch.setattr(tts_service, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("TTS_EXPRESSIVE_ENGINE_ENABLED", "true")
    monkeypatch.setattr(
        tts_service.IndexTTS2ExpressiveEngine,
        "availability",
        lambda self: (False, "test_adapter_unavailable"),
    )

    def fake_generate(text: str, voice: str, speed: float, output_path: Path, *args, **kwargs):
        calls["count"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFfake-wave")

    monkeypatch.setattr(tts_service, "generate_audio", fake_generate)

    response = client.post(
        "/synthesize",
        json={
            "agent": "miss_ciel",
            "text": "Nice work! You said that clearly.",
            "engine": "index_tts2",
            "expressive": True,
            "cache": False,
        },
    )

    assert response.status_code == 200
    assert calls["count"] == 2
    assert response.headers["x-readirect-tts-provider"] == "kokoro"
    assert response.headers["x-readirect-tts-requested-engine"] == "index_tts2"
    assert response.headers["x-readirect-tts-fallback-reason"] == "test_adapter_unavailable"


def test_text_humanizer_preserves_protected_letters_words_and_choices():
    profile = tts_service.AGENT_PROFILES["miss_ciel"]

    for text in ["A", "cat", "A. cat B. dog"]:
        humanized, synthesis_text, text_result, delivery_result, _ = tts_service.prepare_tts_text(
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

    humanized, synthesis_text, text_result, _, _ = tts_service.prepare_tts_text("miss_ciel", "Good job.", profile, request)

    assert humanized == "Good job."
    assert synthesis_text == "Good job."
    assert text_result.applied is False


def test_delivery_control_can_be_disabled():
    profile = tts_service.AGENT_PROFILES["miss_vivian"]
    request = tts_service.SynthesizeRequest(agent="miss_vivian", text="When you're ready say it", humanize=False, delivery_control=False)

    _, synthesis_text, _, delivery_result, _ = tts_service.prepare_tts_text("miss_vivian", "When you're ready say it", profile, request)

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
    assert '"humanized_text": "Nice work! You said that clearly, and I can hear that you\'re getting more confident."' in caplog.text
    assert '"detected_intent": "happy_praise"' in caplog.text
