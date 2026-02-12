import tempfile
from pathlib import Path

import yaml

from yt_transcribe.output import sanitize, VideoMeta, VideoOutputDir, get_transcribed_ids


class TestSanitize:
    def test_lowercase(self):
        assert sanitize("Hello World") == "hello_world"

    def test_special_chars(self):
        assert sanitize("Test! @#$% Video") == "test_video"

    def test_collapse_underscores(self):
        assert sanitize("a---b___c") == "a_b_c"

    def test_strip_edges(self):
        assert sanitize("___hello___") == "hello"

    def test_empty(self):
        assert sanitize("!!!") == ""

    def test_numbers_preserved(self):
        assert sanitize("Video 123 Test") == "video_123_test"

    def test_unicode(self):
        assert sanitize("Über Straße") == "ber_stra_e"

    def test_already_clean(self):
        assert sanitize("hello_world") == "hello_world"


class TestVideoOutputDir:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            meta = VideoMeta(title="My Video", id="abc123", channel="My Channel", url="https://example.com")
            out = VideoOutputDir(base, meta)
            assert out.path.is_dir()
            assert out.path == base / "my_channel" / "my_video"
            assert out.basename == "my_video"

    def test_write_info_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            meta = VideoMeta(title="Test", id="xyz", channel="Chan", url="https://example.com")
            out = VideoOutputDir(base, meta)
            out.write_info()
            data = yaml.safe_load(out.info_path.read_text())
            assert data["title"] == "Test"
            assert data["id"] == "xyz"
            assert data["channel"] == "Chan"
            assert data["url"] == "https://example.com"
            assert "word_count" not in data
            assert "transcribed_at" not in data

    def test_write_info_with_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            meta = VideoMeta(title="Test", id="xyz", channel="Chan", url="https://example.com")
            out = VideoOutputDir(base, meta)
            out.write_info(word_count=500, transcribed_at="2026-01-01T00:00:00Z")
            data = yaml.safe_load(out.info_path.read_text())
            assert data["word_count"] == 500
            assert data["transcribed_at"] == "2026-01-01T00:00:00Z"

    def test_unknown_channel_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            meta = VideoMeta(title="Test", id="xyz", channel="!!!", url="https://example.com")
            out = VideoOutputDir(base, meta)
            assert "unknown_channel" in str(out.path)

    def test_empty_title_uses_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            meta = VideoMeta(title="!!!", id="abc123", channel="Chan", url="https://example.com")
            out = VideoOutputDir(base, meta)
            assert out.basename == "abc123"


class TestGetTranscribedIds:
    def test_reads_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video_dir = base / "video1"
            video_dir.mkdir()
            (video_dir / "info.yaml").write_text(yaml.dump({"id": "abc123", "title": "Test"}))
            ids = get_transcribed_ids(base)
            assert ids == {"abc123"}

    def test_reads_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            video_dir = base / "video1"
            video_dir.mkdir()
            (video_dir / "info.txt").write_text('id: "def456"\ntitle: "Test"\n')
            ids = get_transcribed_ids(base)
            assert ids == {"def456"}

    def test_reads_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d1 = base / "v1"
            d1.mkdir()
            (d1 / "info.yaml").write_text(yaml.dump({"id": "aaa"}))
            d2 = base / "v2"
            d2.mkdir()
            (d2 / "info.txt").write_text('id: "bbb"\n')
            ids = get_transcribed_ids(base)
            assert ids == {"aaa", "bbb"}

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = get_transcribed_ids(Path(tmp))
            assert ids == set()

    def test_nonexistent_dir(self):
        ids = get_transcribed_ids(Path("/nonexistent/path"))
        assert ids == set()
