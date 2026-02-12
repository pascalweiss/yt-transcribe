import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from yt_transcribe.log import log_error


class WhisperError(Exception):
    pass


@dataclass
class TranscriptionResult:
    text: str
    word_count: int
    output_path: Path


class WhisperBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        output_file_base: Path,
        model_path: Path,
        language: str,
        output_format: str,
        use_gpu: bool,
    ) -> TranscriptionResult: ...


class PyWhisperCppBackend:
    """In-process whisper-cpp via pywhispercpp bindings.

    Note: pywhispercpp always uses Metal GPU on Apple Silicon when compiled
    with Metal support. The use_gpu flag is ignored — use WhisperCliBackend
    with --no-gpu if you need to disable GPU.
    """

    def transcribe(
        self,
        audio_path: Path,
        output_file_base: Path,
        model_path: Path,
        language: str,
        output_format: str,
        use_gpu: bool,
    ) -> TranscriptionResult:
        from pywhispercpp.model import Model

        lang = language if language != "auto" else ""
        model = Model(
            str(model_path),
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
            language=lang,
        )
        segments = model.transcribe(str(audio_path))

        full_text = " ".join(seg.text.strip() for seg in segments).strip()
        word_count = len(full_text.split())

        formats = [output_format] if output_format != "all" else ["txt", "vtt", "srt", "csv", "json"]
        primary_output = None

        for fmt in formats:
            out_path = output_file_base.with_suffix(f".{fmt}")
            if fmt == "txt":
                out_path.write_text(full_text + "\n")
            elif fmt == "vtt":
                _write_vtt(segments, out_path)
            elif fmt == "srt":
                _write_srt(segments, out_path)
            elif fmt == "csv":
                _write_csv(segments, out_path)
            elif fmt == "json":
                import json
                data = [{"start": s.t0, "end": s.t1, "text": s.text.strip()} for s in segments]
                out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            if primary_output is None:
                primary_output = out_path

        return TranscriptionResult(text=full_text, word_count=word_count, output_path=primary_output)


def _format_ts(cs: int) -> str:
    """Format centiseconds to HH:MM:SS.mmm."""
    ms = cs * 10
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_ts_srt(cs: int) -> str:
    """Format centiseconds to HH:MM:SS,mmm (SRT style)."""
    return _format_ts(cs).replace(".", ",")


def _write_vtt(segments: list, path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_ts(seg.t0)} --> {_format_ts(seg.t1)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines))


def _write_srt(segments: list, path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_ts_srt(seg.t0)} --> {_format_ts_srt(seg.t1)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines))


def _write_csv(segments: list, path: Path) -> None:
    lines = ["start,end,text"]
    for seg in segments:
        text = seg.text.strip().replace('"', '""')
        lines.append(f"{seg.t0},{seg.t1},\"{text}\"")
    path.write_text("\n".join(lines) + "\n")


class WhisperCliBackend:
    """Subprocess-based whisper-cli backend."""

    def __init__(self, cli_path: str = "whisper-cli") -> None:
        self.cli_path = cli_path

    def transcribe(
        self,
        audio_path: Path,
        output_file_base: Path,
        model_path: Path,
        language: str,
        output_format: str,
        use_gpu: bool,
    ) -> TranscriptionResult:
        args = [
            self.cli_path,
            "--model", str(model_path),
            "--language", language,
            "--no-prints",
            "--output-file", str(output_file_base),
            "--file", str(audio_path),
        ]

        format_flags = {
            "txt": ["--output-txt"],
            "vtt": ["--output-vtt"],
            "srt": ["--output-srt"],
            "csv": ["--output-csv"],
            "json": ["--output-json"],
            "all": ["--output-txt", "--output-vtt", "--output-srt", "--output-csv", "--output-json"],
        }
        args.extend(format_flags.get(output_format, ["--output-txt"]))

        if not use_gpu:
            args.append("--no-gpu")

        whisper_log = output_file_base.parent / "whisper.log"

        with open(whisper_log, "w") as log_file:
            result = subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            log_error(f"Transcription failed (see {whisper_log})")
            raise WhisperError(f"whisper-cli exited with code {result.returncode}")

        whisper_log.unlink(missing_ok=True)

        txt_path = output_file_base.with_suffix(".txt")
        word_count = 0
        text = ""
        if txt_path.is_file():
            text = txt_path.read_text()
            word_count = len(text.split())

        primary = output_file_base.with_suffix(f".{output_format}" if output_format != "all" else ".txt")
        return TranscriptionResult(text=text, word_count=word_count, output_path=primary)


def get_backend(use_gpu: bool = True) -> WhisperBackend:
    """Select the whisper backend.

    Uses pywhispercpp (in-process, bundled) by default.
    Falls back to whisper-cli subprocess if available.

    If use_gpu=False, prefer whisper-cli (which supports --no-gpu) since
    pywhispercpp has no runtime GPU toggle.
    """
    if not use_gpu and shutil.which("whisper-cli"):
        return WhisperCliBackend()
    if not use_gpu:
        from yt_transcribe.log import log
        log("[yellow]Warning:[/yellow] --no-gpu ignored (pywhispercpp has no GPU toggle, whisper-cli not found)")
    return PyWhisperCppBackend()
