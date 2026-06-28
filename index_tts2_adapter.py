from __future__ import annotations

import importlib
import inspect
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import tts_config


SERVICE_ROOT = Path(__file__).resolve().parent
_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_STATUS: Optional["AdapterStatus"] = None


@dataclass(frozen=True)
class AdapterStatus:
    available: bool
    reason: str
    mode: str
    model_dir: str
    config_path: str
    device: str
    package: str = "indextts.infer_v2"
    details: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndexTTS2AdapterError(RuntimeError):
    def __init__(self, reason: str, details: Optional[str] = None) -> None:
        super().__init__(reason if details is None else f"{reason}:{details}")
        self.reason = reason
        self.details = details


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else SERVICE_ROOT / path


def _runtime_status(available: bool, reason: str, mode: str = "unavailable", details: Optional[str] = None) -> AdapterStatus:
    return AdapterStatus(
        available=available,
        reason=reason,
        mode=mode,
        model_dir=str(tts_config.index_tts2_model_dir()),
        config_path=str(tts_config.index_tts2_config_path()),
        device=tts_config.index_tts2_device(),
        details=details,
    )


def _add_repo_path() -> None:
    repo_path = tts_config.index_tts2_repo_path()
    if not repo_path:
        return

    resolved = _resolve_path(Path(repo_path))
    if resolved.exists():
        path_text = str(resolved)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _import_index_tts2():
    _add_repo_path()
    try:
        module = importlib.import_module("indextts.infer_v2")
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        if missing == "indextts" or missing.startswith("indextts."):
            raise IndexTTS2AdapterError("package_missing", "indextts.infer_v2") from exc
        raise IndexTTS2AdapterError("dependency_missing", missing) from exc
    except Exception as exc:
        raise IndexTTS2AdapterError("adapter_import_failed", exc.__class__.__name__) from exc

    tts_class = getattr(module, "IndexTTS2", None)
    if tts_class is None:
        raise IndexTTS2AdapterError("unsupported_api_shape", "IndexTTS2 class missing")

    return tts_class


def _validate_paths() -> tuple[Path, Path]:
    model_dir = tts_config.index_tts2_model_dir()
    config_path = tts_config.index_tts2_config_path()

    if not model_dir.exists():
        raise IndexTTS2AdapterError("model_dir_missing", str(model_dir))
    if not model_dir.is_dir():
        raise IndexTTS2AdapterError("model_dir_invalid", str(model_dir))
    if not config_path.exists():
        raise IndexTTS2AdapterError("config_path_missing", str(config_path))
    if not config_path.is_file():
        raise IndexTTS2AdapterError("config_path_invalid", str(config_path))

    return model_dir, config_path


def _use_cuda() -> bool:
    device = tts_config.index_tts2_device()
    if device == "cpu":
        return False
    if device in {"cuda", "gpu"}:
        return True

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def status() -> dict[str, Any]:
    if not tts_config.index_tts2_enabled():
        return _runtime_status(False, "indextts2_disabled").to_dict()

    try:
        _import_index_tts2()
        _validate_paths()
    except IndexTTS2AdapterError as exc:
        return _runtime_status(False, exc.reason, details=exc.details).to_dict()

    return _runtime_status(True, "available", mode="dual_reference").to_dict()


def _load_model():
    global _MODEL, _MODEL_STATUS

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if not tts_config.index_tts2_enabled():
            raise IndexTTS2AdapterError("indextts2_disabled")

        tts_class = _import_index_tts2()
        model_dir, config_path = _validate_paths()
        use_cuda = _use_cuda()

        try:
            _MODEL = tts_class(
                cfg_path=str(config_path),
                model_dir=str(model_dir),
                use_fp16=use_cuda,
                use_cuda_kernel=False,
                use_deepspeed=False,
            )
        except TypeError:
            try:
                _MODEL = tts_class(cfg_path=str(config_path), model_dir=str(model_dir))
            except Exception as exc:
                raise IndexTTS2AdapterError("model_load_failed", exc.__class__.__name__) from exc
        except Exception as exc:
            raise IndexTTS2AdapterError("model_load_failed", exc.__class__.__name__) from exc

        _MODEL_STATUS = _runtime_status(True, "available", mode="dual_reference")
        return _MODEL


def _infer_kwargs(model: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], str]:
    infer = getattr(model, "infer", None)
    if not callable(infer):
        raise IndexTTS2AdapterError("unsupported_api_shape", "infer callable missing")

    parameters = inspect.signature(infer).parameters
    supported = set(parameters.keys())

    mode = "single_reference"
    filtered = {
        "spk_audio_prompt": kwargs["spk_audio_prompt"],
        "text": kwargs["text"],
        "output_path": kwargs["output_path"],
    }

    has_audio_style_reference = bool(kwargs.get("emo_audio_prompt"))

    if "emo_audio_prompt" in supported and has_audio_style_reference:
        filtered["emo_audio_prompt"] = kwargs["emo_audio_prompt"]
        mode = "dual_reference"

    if "emo_alpha" in supported:
        filtered["emo_alpha"] = kwargs["emo_alpha"]

    if not has_audio_style_reference and "use_emo_text" in supported and kwargs.get("emotion_prompt"):
        filtered["use_emo_text"] = True
        mode = "text_prompt_only"
    if not has_audio_style_reference and "emo_text" in supported and kwargs.get("emotion_prompt"):
        filtered["emo_text"] = kwargs["emotion_prompt"]

    if "use_random" in supported:
        filtered["use_random"] = kwargs["use_random"]
    if "verbose" in supported:
        filtered["verbose"] = kwargs["verbose"]

    return filtered, mode


def generate(
    text: str,
    output_path: str,
    agent: str,
    kokoro_voice: str,
    speed: float,
    speaker_reference_path: Optional[str],
    style_reference_path: Optional[str],
    emotion_prompt: Optional[str],
    intent: str,
) -> dict[str, Any]:
    if kokoro_voice != "af_heart" and agent in {"miss_ciel", "ciel"}:
        raise IndexTTS2AdapterError("voice_identity_mismatch", kokoro_voice)

    if not speaker_reference_path:
        raise IndexTTS2AdapterError("speaker_reference_missing")
    if not style_reference_path:
        raise IndexTTS2AdapterError("style_reference_missing")

    speaker_ref = Path(speaker_reference_path)
    style_ref = Path(style_reference_path)
    if not speaker_ref.exists():
        raise IndexTTS2AdapterError("speaker_reference_missing", str(speaker_ref))
    if not style_ref.exists():
        raise IndexTTS2AdapterError("style_reference_missing", str(style_ref))

    model = _load_model()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs, mode = _infer_kwargs(
        model,
        {
            "spk_audio_prompt": str(speaker_ref),
            "text": text,
            "output_path": str(output),
            "emo_audio_prompt": str(style_ref),
            "emo_alpha": tts_config.index_tts2_emo_alpha(),
            "emotion_prompt": emotion_prompt,
            "use_random": tts_config.index_tts2_use_random(),
            "verbose": tts_config.index_tts2_verbose(),
        },
    )

    try:
        model.infer(**kwargs)
    except IndexTTS2AdapterError:
        raise
    except Exception as exc:
        name = exc.__class__.__name__
        if "cuda" in str(exc).lower() or "cuda" in name.lower():
            raise IndexTTS2AdapterError("device_cuda_error", name) from exc
        raise IndexTTS2AdapterError("generation_failed", name) from exc

    if not output.exists():
        raise IndexTTS2AdapterError("expressive_output_missing", str(output))

    return {
        "output_path": str(output),
        "mode": mode,
        "speaker_reference_path": str(speaker_ref),
        "style_reference_path": str(style_ref),
        "emotion_prompt": emotion_prompt,
        "intent": intent,
        "sample_rate": tts_config.index_tts2_output_sample_rate(),
    }
