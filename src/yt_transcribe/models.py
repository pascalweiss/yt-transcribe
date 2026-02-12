import urllib.request
from pathlib import Path

from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn

from yt_transcribe.log import log, log_error

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "whisper-cpp"
HF_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def resolve_model(name_or_path: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Resolve a model name or file path to an actual GGML model file.

    If name_or_path is an existing file, use it directly.
    Otherwise treat it as a model name, check the cache, and auto-download if needed.
    """
    path = Path(name_or_path)
    if path.is_file():
        return path

    model_file = f"ggml-{name_or_path}.bin"
    model_path = model_dir / model_file

    if model_path.is_file():
        return model_path

    url = f"{HF_BASE_URL}/{model_file}"
    log(f"Downloading model {model_file}...")
    model_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = model_path.with_suffix(".bin.part")
    try:
        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task("Downloading", total=total)
            with open(tmp_path, "wb") as f:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
                    progress.advance(task, len(chunk))

        tmp_path.rename(model_path)
        log(f"Model saved to {model_path}")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        log_error(f"Failed to download model from {url}")
        log_error("Available models: tiny, base, small, medium, large-v3, large-v3-turbo")
        raise SystemExit(1) from e

    return model_path
