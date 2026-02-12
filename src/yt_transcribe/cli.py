import os
from pathlib import Path

import click

from yt_transcribe import __version__
from yt_transcribe.channel import run_channel_mode
from yt_transcribe.log import log_error
from yt_transcribe.models import DEFAULT_MODEL, DEFAULT_MODEL_DIR, resolve_model
from yt_transcribe.transcriber import TranscribeConfig, transcribe_single
from yt_transcribe.whisper import get_backend, WhisperError


@click.command()
@click.version_option(version=__version__, prog_name="yt-transcribe")
@click.argument("url", required=False)
@click.option("-m", "--model", "model_name", default=DEFAULT_MODEL, show_default=True, help="Model name or path to GGML .bin file.")
@click.option("--no-gpu", is_flag=True, default=False, help="Disable GPU acceleration (Metal).")
@click.option("-l", "--language", default="auto", show_default=True, help="Language code (e.g. en, de, auto).")
@click.option("-f", "--output-format", type=click.Choice(["txt", "vtt", "srt", "csv", "json", "all"]), default="txt", show_default=True, help="Transcript output format.")
@click.option("-a", "--audio-format", default="mp3", show_default=True, help="Audio download format.")
@click.option("-k", "--keep-audio", is_flag=True, default=False, help="Keep audio file after transcription.")
@click.option("-o", "--output-dir", type=click.Path(), default=None, help="Base output directory. [default: $YT_TRANSCRIBE_OUTPUT_DIR or .]")
@click.option("-c", "--channel", "channel_url", default=None, help="Channel URL (mutually exclusive with positional URL).")
@click.option("--min-seconds", type=int, default=60, show_default=True, help="Skip videos shorter than N seconds.")
@click.option("--amount", type=int, default=0, show_default=True, help="Max new videos to transcribe (0 = all).")
@click.option("--workers", type=int, default=1, show_default=True, help="Parallel workers for channel mode.")
def main(
    url: str | None,
    model_name: str,
    no_gpu: bool,
    language: str,
    output_format: str,
    audio_format: str,
    keep_audio: bool,
    output_dir: str | None,
    channel_url: str | None,
    min_seconds: int,
    amount: int,
    workers: int,
) -> None:
    """Download and transcribe YouTube videos using yt-dlp and whisper-cpp.

    Provide a video URL as a positional argument, or use --channel for batch mode.
    """
    # Validate mutual exclusivity
    if url and channel_url:
        raise click.UsageError("Cannot use both a positional URL and --channel.")
    if not url and not channel_url:
        raise click.UsageError("Provide a video URL or use --channel <url>.")

    # Resolve output dir
    if output_dir is None:
        output_dir = os.environ.get("YT_TRANSCRIBE_OUTPUT_DIR", ".")
    output_path = Path(output_dir).expanduser().resolve()

    # Get whisper backend (check before downloading model)
    try:
        backend = get_backend(use_gpu=not no_gpu)
    except WhisperError as e:
        log_error(str(e))
        raise SystemExit(1)

    # Resolve model
    model_dir_env = os.environ.get("WHISPER_CPP_MODEL_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else DEFAULT_MODEL_DIR
    model_path = resolve_model(model_name, model_dir)

    config = TranscribeConfig(
        model_path=model_path,
        language=language,
        output_format=output_format,
        audio_format=audio_format,
        keep_audio=keep_audio,
        output_dir=output_path,
        use_gpu=not no_gpu,
    )

    if channel_url:
        results = run_channel_mode(
            channel_url=channel_url,
            config=config,
            backend=backend,
            min_seconds=min_seconds,
            amount=amount,
            workers=workers,
        )
        if any(not r.success for r in results):
            raise SystemExit(1)
    else:
        result = transcribe_single(url, config, backend)
        if not result.success:
            raise SystemExit(1)
