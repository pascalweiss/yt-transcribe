from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from yt_transcribe.downloader import fetch_video_info
from yt_transcribe.downloader import download_audio
from yt_transcribe.log import log, log_step, log_done, log_error
from yt_transcribe.output import VideoMeta, VideoOutputDir
from yt_transcribe.whisper import WhisperBackend, WhisperError


@dataclass
class TranscribeConfig:
    model_path: Path
    language: str
    output_format: str
    audio_format: str
    keep_audio: bool
    output_dir: Path
    use_gpu: bool


@dataclass
class TranscribeResult:
    success: bool
    video_id: str = ""
    title: str = ""
    error: str = ""


def transcribe_single(
    url: str,
    config: TranscribeConfig,
    backend: WhisperBackend,
    worker_id: int = 0,
    progress_label: str = "",
) -> TranscribeResult:
    """Transcribe a single video. Never raises — returns TranscribeResult."""
    prefix = f"({progress_label}) " if progress_label else ""

    try:
        info = fetch_video_info(url)
    except Exception as e:
        log_error(f"{prefix}Failed to fetch metadata for {url}: {e}", worker=worker_id)
        return TranscribeResult(success=False, error=str(e))

    log(f"{prefix}Processing: \"{info.title}\" ({info.id})", worker=worker_id)

    try:
        meta = VideoMeta(title=info.title, id=info.id, channel=info.channel, url=url)
        out = VideoOutputDir(config.output_dir, meta)

        out.write_info()

        log_step("download", "Downloading audio...", worker=worker_id)
        audio_path = download_audio(url, out.path, out.basename, config.audio_format)

        model_name = config.model_path.stem
        gpu_label = "gpu" if config.use_gpu else "cpu"
        log_step("transcribe", f"Transcribing with {model_name} ({gpu_label})...", worker=worker_id)

        result = backend.transcribe(
            audio_path=audio_path,
            output_file_base=out.path / out.basename,
            model_path=config.model_path,
            language=config.language,
            output_format=config.output_format,
            use_gpu=config.use_gpu,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.write_info(word_count=result.word_count, transcribed_at=timestamp)

        if not config.keep_audio:
            audio_path.unlink(missing_ok=True)

        log_done(f"{prefix}{out.path}", worker=worker_id)
        return TranscribeResult(success=True, video_id=info.id, title=info.title)

    except WhisperError as e:
        log_error(f"{prefix}Transcription failed: {e}", worker=worker_id)
        return TranscribeResult(success=False, video_id=info.id, title=info.title, error=str(e))
    except Exception as e:
        log_error(f"{prefix}Failed: {e}", worker=worker_id)
        return TranscribeResult(success=False, video_id=info.id, title=info.title, error=str(e))
