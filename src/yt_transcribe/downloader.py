from dataclasses import dataclass
from pathlib import Path

import yt_dlp


@dataclass
class VideoInfo:
    title: str
    id: str
    channel: str
    url: str


def fetch_video_info(url: str) -> VideoInfo:
    """Fetch video metadata without downloading."""
    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return VideoInfo(
        title=info.get("title", ""),
        id=info.get("id", ""),
        channel=info.get("channel", "") or info.get("uploader", ""),
        url=url,
    )


def download_audio(url: str, output_dir: Path, basename: str, audio_format: str = "mp3") -> Path:
    """Download audio from a YouTube video using yt-dlp library.

    Returns the path to the downloaded audio file.
    """
    output_template = str(output_dir / f"{basename}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "extract_audio": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "0",
        }],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return output_dir / f"{basename}.{audio_format}"


def fetch_channel_video_ids(
    channel_url: str,
    min_seconds: int = 60,
    after_ts: int | None = None,
    before_ts: int | None = None,
    tab: str = "videos",
) -> list[str]:
    """Fetch all video IDs from a channel, filtered by minimum duration and optional date range."""
    # Ensure we target the correct tab
    url = channel_url.rstrip("/")
    # Strip any existing tab suffix before appending the desired one
    for suffix in ("/videos", "/shorts", "/streams"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    url += f"/{tab}"

    # Build filter expression
    # Shorts don't provide duration or timestamp metadata in extract_flat mode
    filters: list[str] = []
    if tab != "shorts":
        filters.append(f"duration>={min_seconds}")
        if after_ts is not None:
            filters.append(f"timestamp>={after_ts}")
        if before_ts is not None:
            filters.append(f"timestamp<={before_ts}")

    opts: dict = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
    }

    if filters:
        filter_expr = " & ".join(filters)
        opts["match_filter"] = yt_dlp.utils.match_filter_func(filter_expr)

    # Request approximate timestamps when date filtering is active
    if tab != "shorts" and (after_ts is not None or before_ts is not None):
        opts["extractor_args"] = {"youtubetab": {"approximate_date": [""]}}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries", []) if info else []
    # Filter out non-video IDs (e.g. channel IDs starting with "UC")
    return [
        e["id"] for e in entries
        if e and "id" in e and len(e["id"]) == 11 and not e["id"].startswith("UC")
    ]
