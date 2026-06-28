from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import soundfile as sf


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = SERVICE_ROOT / "storage" / "tts" / "references"
DEFAULT_REPORT_PATH = SERVICE_ROOT / "READIRECT_TTS_REFERENCE_AUDIO_DURATION_REPORT.md"
DEFAULT_SAMPLE_RATE = 24000
MIN_TARGET_SECONDS = 7.0


@dataclass(frozen=True)
class CategorySpec:
    intent: str
    prefixes: tuple[str, ...]
    expected_count: int


AGENT_CATEGORIES: dict[str, tuple[CategorySpec, ...]] = {
    "ciel": (
        CategorySpec("intro", ("intro",), 1),
        CategorySpec("friendly_encouragement", ("friendly", "freindly"), 3),
        CategorySpec("gentle_reassurance", ("gentle",), 3),
        CategorySpec("happy_praise", ("happy",), 3),
        CategorySpec("focused_instruction", ("instruct", "focus"), 3),
        CategorySpec("playful_friend", ("playful",), 3),
    ),
    "vivian": (
        CategorySpec("intro", ("intro",), 1),
        CategorySpec("friendly_encouragement", ("friendly",), 3),
        CategorySpec("gentle_reassurance", ("gentle",), 3),
        CategorySpec("happy_praise", ("happy",), 3),
        CategorySpec("focused_instruction", ("focus", "instruct"), 3),
    ),
    "estelle": (
        CategorySpec("intro", ("intro",), 1),
        CategorySpec("gentle_reassurance", ("gentle",), 3),
        CategorySpec("calm_evaluation", ("calm",), 3),
        CategorySpec("happy_praise", ("happy",), 3),
        CategorySpec("focused_instruction", ("focus", "instruct"), 3),
    ),
}


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 1


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def source_mp3s(agent_root: Path, intent: str, prefixes: Sequence[str]) -> list[Path]:
    lowered = tuple(prefix.lower() for prefix in prefixes)
    candidates = list(agent_root.glob("*.mp3")) + list((agent_root / intent).glob("*.mp3"))
    matches = [path for path in candidates if any(path.stem.lower().startswith(prefix) for prefix in lowered)]
    return sorted(matches, key=lambda path: (numeric_suffix(path), path.name.lower()))


def convert_with_ffmpeg(source: Path, destination: Path, sample_rate: int, ffmpeg: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        "loudnorm=I=-20:TP=-1.5:LRA=11",
        str(destination),
    ]
    subprocess.run(command, check=True)


def validate_wav(path: Path, sample_rate: int) -> None:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise RuntimeError(f"{path} is not mono.")
        if handle.getframerate() != sample_rate:
            raise RuntimeError(f"{path} is not {sample_rate} Hz.")


def convert_mp3(source: Path, destination: Path, sample_rate: int, dry_run: bool = False) -> str:
    if dry_run:
        return "dry_run"

    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for MP3 reference conversion.")

    convert_with_ffmpeg(source, destination, sample_rate, ffmpeg)
    validate_wav(destination, sample_rate)
    return "ffmpeg"


def clean_agent_wavs(agent_root: Path, categories: Sequence[CategorySpec]) -> int:
    removed = 0
    for spec in categories:
        category_root = agent_root / spec.intent
        if not category_root.exists():
            continue
        for path in category_root.glob(f"{agent_root.name}_*.wav"):
            path.unlink()
            removed += 1
    return removed


def move_source_to_category(source: Path, category_root: Path, dry_run: bool) -> None:
    destination = category_root / source.name
    if source.resolve() == destination.resolve():
        return
    print(f"  source mp3 -> {destination}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def organize_agent(reference_root: Path, agent: str, sample_rate: int, clean: bool, move_sources: bool, dry_run: bool) -> int:
    categories = AGENT_CATEGORIES[agent]
    agent_root = reference_root / agent
    if not agent_root.exists():
        print(f"WARNING: reference root missing for {agent}: {agent_root}")
        return 0

    for spec in categories:
        (agent_root / spec.intent).mkdir(parents=True, exist_ok=True)

    if clean and not dry_run:
        removed = clean_agent_wavs(agent_root, categories)
        if removed:
            print(f"Removed {removed} existing generated WAV file(s) for {agent}.")

    converted = 0
    for spec in categories:
        sources = source_mp3s(agent_root, spec.intent, spec.prefixes)
        if not sources:
            print(f"WARNING: no loose MP3 files found for {agent}/{spec.intent}.")
            continue

        if len(sources) < spec.expected_count:
            print(f"WARNING: found {len(sources)} {agent}/{spec.intent} MP3 file(s), expected {spec.expected_count}.")

        if spec.intent == "intro":
            sources = sources[:1]

        for index, source in enumerate(sources, start=1):
            destination = agent_root / spec.intent / f"{agent}_{spec.intent}_{index:02d}.wav"
            print(f"{source.relative_to(reference_root)} -> {destination.relative_to(reference_root)}")
            try:
                method = convert_mp3(source, destination, sample_rate, dry_run=dry_run)
            except Exception as exc:
                print(f"WARNING: failed to convert {source.name}: {exc}", file=sys.stderr)
                continue
            converted += 1
            if method != "dry_run":
                print(f"  converted with {method}")
            if move_sources:
                move_source_to_category(source, destination.parent, dry_run=dry_run)

    return converted


def manifest_entries(reference_root: Path, agent: str, spec: CategorySpec) -> list[str]:
    folder = reference_root / agent / spec.intent
    entries = [f"{agent}/{spec.intent}/{path.name}" for path in sorted(folder.glob(f"{agent}_*.wav"))]
    return entries[:1] if spec.intent == "intro" else entries


def write_manifest(reference_root: Path, dry_run: bool = False) -> dict[str, dict[str, list[str]]]:
    payload: dict[str, dict[str, list[str]]] = {}
    for agent, categories in AGENT_CATEGORIES.items():
        agent_payload: dict[str, list[str]] = {}
        for spec in categories:
            entries = manifest_entries(reference_root, agent, spec)
            if entries:
                agent_payload[spec.intent] = entries
        if agent_payload:
            payload[agent] = agent_payload

    if dry_run:
        print(json.dumps(payload, indent=2))
        return payload

    manifest_path = reference_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {manifest_path}")
    return payload


def duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        return float(sf.info(path).duration)

    probe = ffprobe_path()
    if not probe:
        return float("nan")

    result = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def file_audio_info(path: Path) -> tuple[int | str, int | str]:
    if path.suffix.lower() == ".wav":
        info = sf.info(path)
        return int(info.samplerate), int(info.channels)
    return "source", "source"


def report_rows(reference_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for agent, categories in AGENT_CATEGORIES.items():
        for spec in categories:
            folder = reference_root / agent / spec.intent
            for path in sorted(folder.glob(f"{agent}_*.wav")):
                duration = duration_seconds(path)
                sample_rate, channels = file_audio_info(path)
                rows.append(
                    {
                        "agent": agent,
                        "intent": spec.intent,
                        "file": path.relative_to(reference_root).as_posix(),
                        "duration": f"{duration:.2f}",
                        "status": "OK" if duration >= MIN_TARGET_SECONDS else "SHORT",
                        "sample_rate": str(sample_rate),
                        "channels": str(channels),
                    }
                )
    return rows


def source_rows(reference_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for agent in AGENT_CATEGORIES:
        agent_root = reference_root / agent
        for path in sorted(agent_root.rglob("*.mp3"), key=lambda item: item.relative_to(reference_root).as_posix().lower()):
            duration = duration_seconds(path)
            rows.append(
                {
                    "agent": agent,
                    "file": path.relative_to(reference_root).as_posix(),
                    "duration": f"{duration:.2f}",
                    "status": "OK" if duration >= MIN_TARGET_SECONDS else "SHORT",
                }
            )
    return rows


def write_report(reference_root: Path, report_path: Path) -> None:
    rows = report_rows(reference_root)
    sources = source_rows(reference_root)
    short_count = sum(1 for row in rows if row["status"] == "SHORT")

    lines = [
        "# ReaDirect TTS Reference Audio Duration Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Target for rewritten ReaDirect narration lines: at least 7.00 seconds. Longer is acceptable when the line remains natural.",
        "",
        f"Reference WAV files checked: {len(rows)}",
        f"Reference WAV files below target: {short_count}",
        "",
        "## Converted Reference WAVs",
        "",
        "| Agent | Intent | File | Duration | Status | Sample Rate | Channels |",
        "|---|---|---|---:|---|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['intent']} | `{row['file']}` | {row['duration']}s | {row['status']} | {row['sample_rate']} | {row['channels']} |"
        )

    lines.extend(
        [
            "",
            "## Source MP3s",
            "",
            "| Agent | Source File | Duration | Status |",
            "|---|---|---:|---|",
        ]
    )
    for row in sources:
        lines.append(f"| {row['agent']} | `{row['file']}` | {row['duration']}s | {row['status']} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- WAV references are generated as mono 24 kHz files.",
            "- Conversion preserves the full MP3 length; no silence trimming is applied.",
            "- `SHORT` means the reference is under the 7-second target and should be replaced with a longer prompt take if you want stronger prosody guidance.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize ReaDirect TTS reference MP3s and write a duration report.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--agents", nargs="+", default=["ciel", "vivian", "estelle"], choices=sorted(AGENT_CATEGORIES))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--clean", action="store_true", help="Remove generated WAV files for selected agents before conversion.")
    parser.add_argument("--move-sources", action="store_true", help="Move source MP3 files into their category folders after conversion.")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args(list(argv) if argv is not None else None)

    reference_root = Path(args.reference_root)
    if not reference_root.exists():
        print(f"ERROR: reference root does not exist: {reference_root}", file=sys.stderr)
        return 2

    if not args.skip_convert:
        converted = 0
        for agent in args.agents:
            converted += organize_agent(reference_root, agent, args.sample_rate, clean=args.clean, move_sources=args.move_sources, dry_run=args.dry_run)
        source_note = "Source MP3 files were moved into category folders." if args.move_sources else "Source MP3 files were left in their current locations."
        print(f"Done. Converted {converted} file(s). {source_note}")

    write_manifest(reference_root, dry_run=args.dry_run)
    if not args.dry_run:
        write_report(reference_root, Path(args.report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
