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


def fetch_channel_video_ids(channel_url: str, min_seconds: int = 60) -> list[str]:
    """Fetch all video IDs from a channel, filtered by minimum duration."""
    # Ensure we target the /videos tab to avoid live streams and playlists
    url = channel_url.rstrip("/")
    if not url.endswith("/videos"):
        url += "/videos"

    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration>={min_seconds}"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries", []) if info else []
    # Filter out non-video IDs (e.g. channel IDs starting with "UC")
    return [
        e["id"] for e in entries
        if e and "id" in e and len(e["id"]) == 11 and not e["id"].startswith("UC")
    ]
