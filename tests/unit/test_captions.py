"""Unit tests for the .ass renderer in src/ugc_pipeline/steps/caption.py.

All tests are pure Python — no ffmpeg, no network, no model download.
"""

from __future__ import annotations

from ugc_pipeline.steps.caption import _ass_filter_path, _seconds_to_ass_time, render_ass
from pathlib import Path


# ---------------------------------------------------------------------------
# _seconds_to_ass_time
# ---------------------------------------------------------------------------


def test_seconds_to_ass_time_zero() -> None:
    assert _seconds_to_ass_time(0.0) == "0:00:00.00"


def test_seconds_to_ass_time_one_minute() -> None:
    assert _seconds_to_ass_time(60.0) == "0:01:00.00"


def test_seconds_to_ass_time_fractional() -> None:
    # 1.5 s → 0:00:01.50
    result = _seconds_to_ass_time(1.5)
    assert result == "0:00:01.50"


def test_seconds_to_ass_time_over_one_hour() -> None:
    # 3661 s = 1h 1m 1s
    result = _seconds_to_ass_time(3661.0)
    assert result == "1:01:01.00"


# ---------------------------------------------------------------------------
# render_ass — structural checks
# ---------------------------------------------------------------------------


def test_render_ass_contains_required_sections() -> None:
    """Output must contain Script Info, V4+ Styles, and Events sections."""
    ass = render_ass([])
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass


def test_render_ass_play_res() -> None:
    """PlayResX/Y should match arguments."""
    ass = render_ass([], video_w=1080, video_h=1920)
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass


def test_render_ass_empty_segments_no_dialogue() -> None:
    """Empty segments list must not produce any Dialogue lines."""
    ass = render_ass([])
    assert "Dialogue:" not in ass


def test_render_ass_single_segment_produces_dialogue() -> None:
    """A single segment should produce exactly one Dialogue line."""
    segments = [{"start": 0.0, "end": 2.5, "text": "Ciao mondo", "words": []}]
    ass = render_ass(segments)
    dialogue_lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue_lines) == 1


def test_render_ass_dialogue_time_format() -> None:
    """Dialogue timing should use H:MM:SS.cc format."""
    segments = [{"start": 1.0, "end": 3.0, "text": "test", "words": []}]
    ass = render_ass(segments)
    # Should contain the start time 0:00:01.00
    assert "0:00:01.00" in ass
    assert "0:00:03.00" in ass


# ---------------------------------------------------------------------------
# Italian non-ASCII preservation
# ---------------------------------------------------------------------------


def test_render_ass_preserves_italian_non_ascii() -> None:
    """Italian characters (è, à, ò, ù, ì) must be preserved literally."""
    italian_text = "È una bella giornata, àncora una volta"
    segments = [{"start": 0.0, "end": 3.0, "text": italian_text, "words": []}]
    ass = render_ass(segments)
    assert "È" in ass
    assert "à" in ass


def test_render_ass_preserves_all_italian_accents() -> None:
    """Common Italian accented vowels must all survive the render."""
    text = "è à ò ù ì café naïf"
    segments = [{"start": 0.0, "end": 2.0, "text": text, "words": []}]
    ass = render_ass(segments)
    for char in ("è", "à", "ò", "ù", "ì"):
        assert char in ass, f"Character {char!r} not found in rendered .ass"


# ---------------------------------------------------------------------------
# Line wrapping
# ---------------------------------------------------------------------------


def test_render_ass_wraps_long_line() -> None:
    """A segment longer than max_chars_per_line should contain \\N."""
    long_text = "Questo è un testo molto lungo che dovrebbe essere diviso in più righe perché supera il limite"
    segments = [{"start": 0.0, "end": 5.0, "text": long_text, "words": []}]
    ass = render_ass(segments, max_chars_per_line=20)
    # ASS hard line-break is \N (backslash-N in the file)
    assert r"\N" in ass


def test_render_ass_short_line_not_wrapped() -> None:
    """A short segment should not be wrapped."""
    short_text = "Ciao"
    segments = [{"start": 0.0, "end": 1.0, "text": short_text, "words": []}]
    ass = render_ass(segments, max_chars_per_line=42)
    assert r"\N" not in ass


# ---------------------------------------------------------------------------
# _ass_filter_path — Windows path escaping
# ---------------------------------------------------------------------------


def test_ass_filter_path_backslash_to_forward_slash() -> None:
    """Backslashes should be converted to forward slashes."""
    p = Path("C:\\Users\\user\\artifacts\\captions.ass")
    result = _ass_filter_path(p)
    assert "\\" not in result.replace("\\:", "")  # only escaped colon allowed


def test_ass_filter_path_colon_escaped() -> None:
    """Drive-letter colon (C:) must be escaped as \\:."""
    p = Path("C:/Users/user/captions.ass")
    result = _ass_filter_path(p)
    assert "C\\:" in result


def test_ass_filter_path_unix_style_unchanged_colon() -> None:
    """A Unix-style path with no drive letter should have no escaped colon."""
    p = Path("/tmp/captions.ass")
    result = _ass_filter_path(p)
    assert "\\:" not in result
