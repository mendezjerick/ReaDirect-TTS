from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf


SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = SERVICE_ROOT / "storage" / "tts" / "references"
DEFAULT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class CategorySpec:
    intent: str
    source_prefixes: tuple[str, ...]
    expected_count: int


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec("intro", ("intro",), 1),
    CategorySpec("friendly_encouragement", ("friendly", "freindly"), 3),
    CategorySpec("gentle_reassurance", ("gentle",), 3),
    CategorySpec("happy_praise", ("happy",), 3),
    CategorySpec("focused_instruction", ("instruct",), 3),
    CategorySpec("playful_friend", ("playful",), 3),
)


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 1


def loose_mp3s(ciel_root: Path, prefixes: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    lowered_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for path in ciel_root.glob("*.mp3"):
        stem = path.stem.lower()
        if any(stem.startswith(prefix) for prefix in lowered_prefixes):
            matches.append(path)

    return sorted(matches, key=lambda path: (numeric_suffix(path), path.name.lower()))


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def normalize_array(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    processed = np.asarray(audio, dtype=np.float32)
    if not processed.size:
        return processed

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > 0:
        factor = 0.88 / peak if peak > 0.88 else min(1.25, 0.45 / peak) if peak < 0.35 else 1.0
        if abs(factor - 1.0) > 0.01:
            processed = np.asarray(processed * factor, dtype=np.float32)

    return np.clip(processed, -0.98, 0.98)


def convert_with_soundfile(source: Path, destination: Path, sample_rate: int) -> None:
    audio, rate = sf.read(source, dtype="float32")
    if rate != sample_rate and audio.size:
        source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_length = max(1, int(round(audio.shape[0] * (sample_rate / rate))))
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)

    audio = normalize_array(np.asarray(audio, dtype=np.float32))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate)


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


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise RuntimeError(f"{path} is not mono.")
        if handle.getframerate() != DEFAULT_SAMPLE_RATE:
            raise RuntimeError(f"{path} is not {DEFAULT_SAMPLE_RATE} Hz.")


def convert_mp3(source: Path, destination: Path, sample_rate: int, dry_run: bool = False) -> str:
    if dry_run:
        return "dry_run"

    ffmpeg = ffmpeg_path()
    if ffmpeg:
        convert_with_ffmpeg(source, destination, sample_rate, ffmpeg)
        validate_wav(destination)
        return "ffmpeg"

    try:
        convert_with_soundfile(source, destination, sample_rate)
        validate_wav(destination)
        return "soundfile"
    except Exception as exc:
        raise RuntimeError(
            "MP3 conversion requires ffmpeg, or a soundfile/libsndfile build with MP3 decoding. "
            "Install ffmpeg and rerun this script."
        ) from exc


def existing_manifest_entries(ciel_root: Path, intent: str) -> list[str]:
    folder = ciel_root / intent
    entries = []
    for path in sorted(folder.glob("ciel_*.wav")):
        entries.append(f"ciel/{intent}/{path.name}")
    return entries


def write_manifest(reference_root: Path, dry_run: bool = False) -> dict:
    ciel_root = reference_root / "ciel"
    ciel_manifest: dict[str, list[str]] = {}
    for spec in CATEGORIES:
        entries = existing_manifest_entries(ciel_root, spec.intent)
        if spec.intent == "intro":
            entries = entries[:1]
        if entries:
            ciel_manifest[spec.intent] = entries
            if len(entries) < spec.expected_count:
                print(f"WARNING: {spec.intent} has {len(entries)} reference file(s), expected {spec.expected_count}.")

    payload = {"ciel": ciel_manifest}
    manifest_path = reference_root / "manifest.json"
    if dry_run:
        print(json.dumps(payload, indent=2))
        return payload

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {manifest_path}")
    return payload


def ensure_folders(ciel_root: Path) -> None:
    for spec in CATEGORIES:
        (ciel_root / spec.intent).mkdir(parents=True, exist_ok=True)


def organize(reference_root: Path, sample_rate: int, dry_run: bool = False) -> int:
    ciel_root = reference_root / "ciel"
    if not ciel_root.exists():
        print(f"ERROR: Ciel reference root does not exist: {ciel_root}", file=sys.stderr)
        return 2

    ensure_folders(ciel_root)
    converted_count = 0

    for spec in CATEGORIES:
        sources = loose_mp3s(ciel_root, spec.source_prefixes)
        if not sources:
            print(f"WARNING: no loose MP3 files found for {spec.intent}.")
            continue

        if len(sources) < spec.expected_count:
            print(f"WARNING: found {len(sources)} {spec.intent} MP3 file(s), expected {spec.expected_count}.")

        if spec.intent == "intro":
            sources = sources[:1]

        for index, source in enumerate(sources, start=1):
            destination = ciel_root / spec.intent / f"ciel_{spec.intent}_{index:02d}.wav"
            print(f"{source.name} -> {destination.relative_to(reference_root)}")
            try:
                method = convert_mp3(source, destination, sample_rate, dry_run=dry_run)
            except Exception as exc:
                print(f"WARNING: failed to convert {source.name}: {exc}", file=sys.stderr)
                continue
            converted_count += 1
            if method != "dry_run":
                print(f"  converted with {method}")

    write_manifest(reference_root, dry_run=dry_run)
    print(f"Done. Converted {converted_count} file(s). Original MP3 files were left in place.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize Ciel loose MP3 expressive references into normalized WAV category folders.")
    parser.add_argument("--reference-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    return organize(Path(args.reference_root), args.sample_rate, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
