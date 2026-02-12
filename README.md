# yt-transcribe

Download and transcribe YouTube videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp). Supports single videos and full channel batch mode with parallel workers.

## Install

### Homebrew (macOS)

```bash
brew install pascalweiss/tap/yt-transcribe
```

### pip / uv

```bash
pip install yt-transcribe
# or
uv tool install yt-transcribe
```

**Requirements:** `ffmpeg` must be available on your PATH.

## Usage

### Single video

```bash
yt-transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Channel mode

Transcribe all videos from a channel (skips already-transcribed ones):

```bash
yt-transcribe --channel "https://www.youtube.com/@CHANNEL" -o ./transcripts --workers 4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Whisper model name or path to `.bin` file | `base` |
| `--no-gpu` | Disable GPU acceleration (Metal) | off |
| `-l, --language` | Language code (`en`, `de`, `auto`, ...) | `auto` |
| `-f, --output-format` | Output format: `txt`, `vtt`, `srt`, `csv`, `json`, `all` | `txt` |
| `-a, --audio-format` | Audio download format | `mp3` |
| `-k, --keep-audio` | Keep audio file after transcription | off |
| `-o, --output-dir` | Output directory | `$YT_TRANSCRIBE_OUTPUT_DIR` or `.` |
| `-c, --channel` | Channel URL for batch mode | |
| `--min-seconds` | Skip videos shorter than N seconds | `60` |
| `--amount` | Max new videos to transcribe (`0` = all) | `0` |
| `--workers` | Parallel workers for channel mode | `1` |
| `--version` | Show version and exit | |

## License

[MIT](LICENSE)
