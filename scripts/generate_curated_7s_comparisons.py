from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TTS_EXPRESSIVE_ENGINE_ENABLED", "true")
os.environ.setdefault("TTS_EXPRESSIVE_ENGINE", "index_tts2")
os.environ.setdefault("TTS_EXPRESSIVE_FALLBACK_TO_KOKORO", "true")
os.environ.setdefault("INDEXTTS2_ENABLED", "true")
os.environ.setdefault("INDEXTTS2_ADAPTER_MODULE", "index_tts2_adapter")
os.environ.setdefault("INDEXTTS2_REPO_PATH", str(ROOT / "external" / "index-tts"))
os.environ.setdefault("INDEXTTS2_MODEL_DIR", str(ROOT / "external" / "index-tts" / "checkpoints"))
os.environ.setdefault("INDEXTTS2_CONFIG_PATH", str(ROOT / "external" / "index-tts" / "checkpoints" / "config.yaml"))
os.environ.setdefault("INDEXTTS2_DEVICE", "auto")
os.environ.setdefault("INDEXTTS2_OUTPUT_SAMPLE_RATE", "24000")
os.environ.setdefault("TTS_REFERENCE_MANIFEST_PATH", str(ROOT / "storage" / "tts" / "references" / "manifest.json"))
os.environ.setdefault("TTS_REFERENCE_AUDIO_ROOT", str(ROOT / "storage" / "tts" / "references"))
os.environ.setdefault("TTS_CACHE_ROOT", str(ROOT / "storage" / "tts" / "cache"))
os.environ.setdefault("TTS_COMPARISON_OUTPUT_ROOT", str(ROOT / "storage" / "tts" / "cache" / "comparisons"))
os.environ.setdefault("TTS_AUTO_PROMPT_EXTENSION_ENABLED", "false")
os.environ.setdefault("TTS_CURATED_PROMPTS_ENABLED", "true")
os.environ.setdefault("TTS_REFERENCE_WEIGHTING_ENABLED", "true")
os.environ.setdefault("TTS_CACHE_BYPASS", "false")

from audio_humanizer import AudioHumanizerResult
from curated_agent_lines import CuratedLine, lines_for
from tts_humanizer import humanize_text
import tts_config
import tts_service


CASES = (
    ("miss_ciel", "intro"),
    ("miss_ciel", "friendly_encouragement"),
    ("miss_ciel", "gentle_reassurance"),
    ("miss_ciel", "happy_praise"),
    ("miss_ciel", "focused_instruction"),
    ("miss_ciel", "playful_friend"),
    ("miss_vivian", "intro"),
    ("miss_vivian", "friendly_encouragement"),
    ("miss_vivian", "gentle_reassurance"),
    ("miss_vivian", "happy_praise"),
    ("miss_vivian", "focused_instruction"),
    ("miss_estelle", "intro"),
    ("miss_estelle", "calm_evaluation"),
    ("miss_estelle", "gentle_reassurance"),
    ("miss_estelle", "happy_praise"),
    ("miss_estelle", "focused_instruction"),
)


@contextmanager
def temporary_env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def duration(path: Path) -> Optional[float]:
    try:
        return float(sf.info(path).duration)
    except Exception:
        return None


def duration_label(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def within_range(value: Optional[float], line: CuratedLine) -> str:
    if value is None:
        return "unknown"
    return "yes" if line.min_duration_seconds <= value <= line.max_duration_seconds else "no"


def generate_kokoro(text: str, agent: str, output_path: Path, force: bool) -> AudioHumanizerResult:
    profile = tts_service.AGENT_PROFILES[agent]
    if output_path.exists() and not force:
        return AudioHumanizerResult(
            enabled=False,
            normalized=False,
            faded=False,
            pause_control_applied=False,
            breath_applied=False,
            duration_seconds=duration(output_path) or 0.0,
        )
    return tts_service.generate_audio(
        text,
        profile.voice,
        profile.default_speed,
        output_path,
        audio_humanizer_request=False,
        pause_control_request=True,
    )


def old_humanizer_text(agent: str, raw_text: str) -> str:
    profile = tts_service.AGENT_PROFILES[agent]
    with temporary_env(
        {
            "TTS_CURATED_PROMPTS_ENABLED": "false",
            "TTS_AUTO_PROMPT_EXTENSION_ENABLED": "true",
            "TTS_TEXT_HUMANIZER_VARIATION_ENABLED": "false",
        }
    ):
        result = humanize_text(agent, raw_text, profile, context={"context": "agent_narration"})
        delivery = tts_service.apply_delivery_direction(agent, result.text, profile, {"context": "agent_narration"})
        return delivery.text


def expressive_output(line: CuratedLine, raw_text: str, output_path: Path, force: bool) -> dict:
    request = tts_service.SynthesizeRequest(
        agent=line.agent,
        text=raw_text,
        intent=line.intent,
        line_key=line.line_key,
        engine="index_tts2",
        expressive=True,
        cache=not force,
        humanize=True,
        delivery_control=True,
        audio_humanizer=False,
        pause_control=True,
        context="agent_narration",
    )
    prepared = tts_service.audio_path_for(request)
    if force and prepared.output_path.exists():
        prepared.output_path.unlink()

    started = time.perf_counter()
    try:
        if prepared.output_path.exists() and not force:
            engine_result = tts_service._cached_engine_result(
                prepared,
                AudioHumanizerResult(
                    enabled=False,
                    normalized=False,
                    faded=False,
                    pause_control_applied=False,
                    breath_applied=False,
                    duration_seconds=duration(prepared.output_path) or 0.0,
                ),
            )
            cache_hit = True
        else:
            engine_result = tts_service.generate_prepared_audio(prepared, request)
            cache_hit = False
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))

        if prepared.output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared.output_path, output_path)

        return {
            "prepared": prepared,
            "engine_result": engine_result,
            "error": None,
            "cache_hit": cache_hit,
            "duration_ms": elapsed_ms,
        }
    except Exception as exc:
        return {
            "prepared": prepared,
            "engine_result": None,
            "error": f"{exc.__class__.__name__}: {exc}",
            "cache_hit": False,
            "duration_ms": int(round((time.perf_counter() - started) * 1000)),
        }


def case_line(agent: str, intent: str) -> CuratedLine:
    matches = lines_for(agent, intent)
    if not matches:
        raise RuntimeError(f"No curated line configured for {agent}/{intent}")
    return matches[0]


def write_report(rows: list[dict], report_path: Path) -> None:
    lines = [
        "# ReaDirect Curated 7s TTS Generation Report",
        "",
        "Source of selected curated lines: `ReaDirect-TTS/curated_agent_lines.py`.",
        "Raw app-owned lines are still accepted through Laravel/Vue; explicit `intent` and `line_key` now select curated prompts when provided.",
        "",
        "| Agent | Intent | Source repo/file | Line key | Final curated prompt | Reference file | Ref duration | Priority | Weight | Output | Output duration | In range | Fallback | Error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        text = row["line"].text.replace("|", "\\|")
        reference = (row.get("reference") or "").replace("|", "\\|")
        output = row.get("expressive_output") or ""
        lines.append(
            "| {agent} | {intent} | `ReaDirect-TTS/curated_agent_lines.py` | `{line_key}` | {text} | `{reference}` | {ref_duration} | {priority} | {weight} | `{output}` | {out_duration} | {in_range} | {fallback} | {error} |".format(
                agent=row["line"].agent,
                intent=row["line"].intent,
                line_key=row["line"].line_key,
                text=text,
                reference=reference,
                ref_duration=duration_label(row.get("reference_duration")),
                priority=row.get("reference_priority") or "none",
                weight=row.get("reference_weight") or 0,
                output=output,
                out_duration=duration_label(row.get("expressive_duration")),
                in_range=row.get("within_range") or "unknown",
                fallback=row.get("fallback") or "none",
                error=(row.get("error") or "none").replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "Generated files per case:",
            "",
            "- `*_kokoro_raw.wav`: raw base line through Kokoro.",
            "- `*_old_humanizer.wav`: old generic humanizer output, generated only for comparison.",
            "- `*_curated_kokoro.wav`: curated prompt through Kokoro.",
            "- `*_index_tts2_expressive.wav`: IndexTTS2 expressive attempt, or fallback audio when fallback is configured.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate curated 7-second ReaDirect TTS comparison outputs.")
    parser.add_argument("--force", action="store_true", help="Regenerate comparison files and expressive cache entries.")
    args = parser.parse_args()

    output_root = tts_config.comparison_output_root() / "curated_7s"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    counters: dict[tuple[str, str], int] = {}
    for agent, intent in CASES:
        line = case_line(agent, intent)
        raw_text = line.legacy_texts[0] if line.legacy_texts else line.text
        key = (agent, intent)
        counters[key] = counters.get(key, 0) + 1
        stem = f"{agent.replace('miss_', '')}_{intent}_{counters[key]:02d}"

        raw_path = output_root / f"{stem}_kokoro_raw.wav"
        old_path = output_root / f"{stem}_old_humanizer.wav"
        curated_path = output_root / f"{stem}_curated_kokoro.wav"
        expressive_path = output_root / f"{stem}_index_tts2_expressive.wav"

        generate_kokoro(raw_text, agent, raw_path, args.force)
        generate_kokoro(old_humanizer_text(agent, raw_text), agent, old_path, args.force)
        generate_kokoro(line.text, agent, curated_path, args.force)
        expressive = expressive_output(line, raw_text, expressive_path, args.force)

        prepared = expressive["prepared"]
        engine_result = expressive["engine_result"]
        expressive_duration = duration(expressive_path) if expressive_path.exists() else None
        rows.append(
            {
                "line": line,
                "reference": prepared.reference_selection.relative_path,
                "reference_duration": prepared.reference_selection.duration_seconds,
                "reference_priority": prepared.reference_selection.priority,
                "reference_weight": prepared.reference_selection.weight,
                "expressive_output": str(expressive_path.relative_to(ROOT)),
                "expressive_duration": expressive_duration,
                "within_range": within_range(expressive_duration, line),
                "fallback": engine_result.fallback_reason if engine_result else prepared.fallback_reason,
                "error": expressive["error"],
            }
        )

        print(
            f"{agent}/{intent}: ref={prepared.reference_selection.relative_path or 'none'} "
            f"priority={prepared.reference_selection.priority} output={expressive_path.name} "
            f"fallback={rows[-1]['fallback'] or 'none'} error={rows[-1]['error'] or 'none'}"
        )

    write_report(rows, output_root / "GENERATION_REPORT.md")
    print(f"Report written to {output_root / 'GENERATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
