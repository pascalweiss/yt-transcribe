import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def sanitize(name: str) -> str:
    """Sanitize a name to ASCII-safe lowercase with underscores.

    Matches the bash behavior: lowercase, non-alnum to underscore,
    collapse runs, strip leading/trailing underscores.
    """
    s = name.lower()
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


@dataclass
class VideoMeta:
    title: str
    id: str
    channel: str
    url: str


class VideoOutputDir:
    """Manages the output directory for a single video transcription."""

    def __init__(self, base_dir: Path, meta: VideoMeta) -> None:
        self.meta = meta
        ch_dir = sanitize(meta.channel) or "unknown_channel"
        vid_dir = sanitize(meta.title) or sanitize(meta.id)
        self.path = base_dir / ch_dir / vid_dir
        self.path.mkdir(parents=True, exist_ok=True)
        self.info_path = self.path / "info.yaml"
        self.basename = vid_dir

    def write_info(self, word_count: int | None = None, transcribed_at: str | None = None) -> None:
        data: dict = {
            "title": self.meta.title,
            "id": self.meta.id,
            "channel": self.meta.channel,
            "url": self.meta.url,
        }
        if word_count is not None:
            data["word_count"] = word_count
        if transcribed_at is not None:
            data["transcribed_at"] = transcribed_at
        self.info_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))


def get_transcribed_ids(base_dir: Path) -> set[str]:
    """Scan info.yaml and info.txt files under base_dir for video IDs."""
    ids: set[str] = set()
    if not base_dir.is_dir():
        return ids
    for info_file in base_dir.rglob("info.yaml"):
        try:
            data = yaml.safe_load(info_file.read_text())
            if isinstance(data, dict) and "id" in data:
                ids.add(str(data["id"]))
        except Exception:
            pass
    for info_file in base_dir.rglob("info.txt"):
        try:
            for line in info_file.read_text().splitlines():
                if line.startswith("id: "):
                    vid = line[4:].strip().strip('"')
                    if vid:
                        ids.add(vid)
        except Exception:
            pass
    return ids
