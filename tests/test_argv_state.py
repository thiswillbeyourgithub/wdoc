"""Unit tests for wdoc.__main__.ArgvState.

Covers every helper invoked from cli_launcher plus the documented edge cases.
Each test asserts on all three of state.args, state.kwargs, and sys.argv so
that desync between them is caught.
"""

import os
import sys

import pytest

os.environ["OVERRIDE_USER_DIR_PYTEST_WDOC"] = "true"
os.environ["WDOC_TYPECHECKING"] = "crash"

from wdoc.__main__ import ArgvState


def _argv(monkeypatch, *tokens):
    monkeypatch.setattr(sys, "argv", list(tokens))


# ---------- rename_positional ----------


@pytest.mark.basic
def test_rename_positional_summary(monkeypatch):
    _argv(monkeypatch, "wdoc", "summary", "--path=/tmp/x")
    s = ArgvState()
    s.rename_positional("summary", "summarize")
    assert "summarize" in s.args and "summary" not in s.args
    assert "summarize" in sys.argv and "summary" not in sys.argv


@pytest.mark.basic
def test_rename_positional_summary_then_query(monkeypatch):
    _argv(monkeypatch, "wdoc", "summary_then_query", "--path=/tmp/x")
    s = ArgvState()
    s.rename_positional("summary_then_query", "summarize_then_query")
    assert "summarize_then_query" in s.args
    assert "summary_then_query" not in s.args
    assert "summarize_then_query" in sys.argv
    assert "summary_then_query" not in sys.argv


@pytest.mark.basic
def test_rename_positional_noop_when_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "query", "--path=/tmp/x")
    s = ArgvState()
    before_args = list(s.args)
    before_argv = list(sys.argv)
    s.rename_positional("summary", "summarize")
    assert s.args == before_args
    assert sys.argv == before_argv


# ---------- remove_positional ----------


@pytest.mark.basic
def test_remove_positional(monkeypatch):
    _argv(monkeypatch, "wdoc", "web", "query", "--path=/tmp/x")
    s = ArgvState()
    s.remove_positional("web")
    assert "web" not in s.args
    assert "web" not in sys.argv
    # other tokens preserved
    assert "query" in s.args
    assert "--path=/tmp/x" in sys.argv


@pytest.mark.basic
def test_remove_positional_noop_when_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "query", "--path=/tmp/x")
    s = ArgvState()
    before_args = list(s.args)
    before_argv = list(sys.argv)
    s.remove_positional("web")
    assert s.args == before_args
    assert sys.argv == before_argv


# ---------- set_kwarg ----------


@pytest.mark.basic
def test_set_kwarg_appends_when_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "--task=query")
    s = ArgvState()
    s.set_kwarg("path", "https://example.com")
    assert s.kwargs["path"] == "https://example.com"
    assert "--path=https://example.com" in sys.argv


@pytest.mark.basic
def test_set_kwarg_noop_when_present_and_not_forced(monkeypatch):
    _argv(monkeypatch, "wdoc", "--filetype=pdf", "--task=query")
    s = ArgvState()
    s.set_kwarg("filetype", "ddg")
    assert s.kwargs["filetype"] == "pdf"
    assert "--filetype=pdf" in sys.argv
    assert "--filetype=ddg" not in sys.argv


@pytest.mark.basic
def test_set_kwarg_force_overwrites_and_dedupes_argv(monkeypatch):
    # filetype was set to something other than ddg; force=True replaces it
    # without leaving stale --filetype=... tokens behind.
    _argv(monkeypatch, "wdoc", "--filetype=pdf", "--task=query")
    s = ArgvState()
    s.set_kwarg("filetype", "ddg", force=True)
    assert s.kwargs["filetype"] == "ddg"
    assert "--filetype=ddg" in sys.argv
    assert "--filetype=pdf" not in sys.argv
    # only one --filetype=... token remains
    filetype_tokens = [t for t in sys.argv if t.startswith("--filetype")]
    assert len(filetype_tokens) == 1


@pytest.mark.basic
def test_set_kwarg_force_dedupes_bare_flag_form(monkeypatch):
    # If the bare `--filetype` form somehow ends up in sys.argv, force=True
    # should still drop it before appending --filetype=ddg.
    _argv(monkeypatch, "wdoc", "--filetype", "--task=query")
    s = ArgvState()
    s.set_kwarg("filetype", "ddg", force=True)
    assert "--filetype" not in sys.argv
    assert "--filetype=ddg" in sys.argv


# ---------- promote_positional_to_kwarg ----------


@pytest.mark.basic
def test_promote_positional_to_kwarg(monkeypatch):
    _argv(monkeypatch, "wdoc", "query", "--path=/tmp/x")
    s = ArgvState()
    s.promote_positional_to_kwarg("query", "task")
    assert "query" not in s.args
    assert s.kwargs["task"] == "query"
    assert "--task=query" in sys.argv
    assert "query" not in sys.argv


@pytest.mark.basic
def test_promote_positional_noop_when_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "--task=query", "--path=/tmp/x")
    s = ArgvState()
    before_args = list(s.args)
    before_kwargs = dict(s.kwargs)
    before_argv = list(sys.argv)
    s.promote_positional_to_kwarg("query", "task")
    assert s.args == before_args
    assert s.kwargs == before_kwargs
    assert sys.argv == before_argv


# ---------- rename_kwarg ----------


@pytest.mark.basic
def test_rename_kwarg_renames_in_kwargs_and_argv(monkeypatch):
    _argv(monkeypatch, "wdoc", "--ddg_max_result=10", "--task=query")
    s = ArgvState()
    s.rename_kwarg("ddg_max_result", "ddg_max_results")
    assert "ddg_max_result" not in s.kwargs
    assert s.kwargs["ddg_max_results"] == 10
    assert "--ddg_max_results=10" in sys.argv
    assert "--ddg_max_result=10" not in sys.argv


@pytest.mark.basic
def test_rename_kwarg_skips_when_target_exists(monkeypatch):
    _argv(
        monkeypatch,
        "wdoc",
        "--ddg_max_result=10",
        "--ddg_max_results=5",
        "--task=query",
    )
    s = ArgvState()
    s.rename_kwarg("ddg_max_result", "ddg_max_results")
    # both stay untouched because target was already set
    assert s.kwargs["ddg_max_result"] == 10
    assert s.kwargs["ddg_max_results"] == 5
    assert "--ddg_max_result=10" in sys.argv
    assert "--ddg_max_results=5" in sys.argv


@pytest.mark.basic
def test_rename_kwarg_noop_when_source_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "--task=query")
    s = ArgvState()
    before_kwargs = dict(s.kwargs)
    before_argv = list(sys.argv)
    s.rename_kwarg("ddg_max_result", "ddg_max_results")
    assert s.kwargs == before_kwargs
    assert sys.argv == before_argv


@pytest.mark.basic
def test_rename_kwarg_handles_space_separated_form(monkeypatch):
    # fire accepts both --key=value and --key value; the helper has to
    # rewrite the bare flag token in the latter case.
    _argv(monkeypatch, "wdoc", "--yt_lang", "en", "--task=query")
    s = ArgvState()
    s.rename_kwarg("yt_lang", "youtube_lang")
    assert "yt_lang" not in s.kwargs
    assert s.kwargs["youtube_lang"] == "en"
    assert "--youtube_lang" in sys.argv
    assert "--yt_lang" not in sys.argv


# ---------- rename_kwarg_prefix ----------


@pytest.mark.basic
def test_rename_kwarg_prefix_multiple(monkeypatch):
    _argv(
        monkeypatch,
        "wdoc",
        "--yt_lang=en",
        "--yt_audio_format=mp3",
        "--task=query",
    )
    s = ArgvState()
    s.rename_kwarg_prefix("yt_", "youtube_")
    assert "yt_lang" not in s.kwargs
    assert "yt_audio_format" not in s.kwargs
    assert s.kwargs["youtube_lang"] == "en"
    assert s.kwargs["youtube_audio_format"] == "mp3"
    assert "--youtube_lang=en" in sys.argv
    assert "--youtube_audio_format=mp3" in sys.argv
    assert not any(t.startswith("--yt_") for t in sys.argv)


@pytest.mark.basic
def test_rename_kwarg_prefix_leaves_non_matching_alone(monkeypatch):
    # A kwarg that does not start with the prefix must not be touched.
    _argv(monkeypatch, "wdoc", "--yt_lang=en", "--filetype=pdf", "--task=query")
    s = ArgvState()
    s.rename_kwarg_prefix("yt_", "youtube_")
    assert s.kwargs["youtube_lang"] == "en"
    assert s.kwargs["filetype"] == "pdf"
    assert "--filetype=pdf" in sys.argv


# ---------- append_positional ----------


@pytest.mark.basic
def test_append_positional(monkeypatch):
    _argv(monkeypatch, "wdoc", "--task=query")
    s = ArgvState()
    s.append_positional("/tmp/piped.txt")
    assert "/tmp/piped.txt" in s.args
    assert "/tmp/piped.txt" in sys.argv


# ---------- read-only check helpers ----------


@pytest.mark.basic
def test_is_empty_true_for_program_name_only(monkeypatch):
    _argv(monkeypatch, "wdoc")
    s = ArgvState()
    assert s.is_empty()


@pytest.mark.basic
def test_is_empty_false_with_args(monkeypatch):
    _argv(monkeypatch, "wdoc", "query", "--path=/tmp/x")
    s = ArgvState()
    assert not s.is_empty()


@pytest.mark.basic
def test_kwarg_equals(monkeypatch):
    _argv(monkeypatch, "wdoc", "--task=parse", "--path=/tmp/x")
    s = ArgvState()
    assert s.kwarg_equals("task", "parse")
    assert not s.kwarg_equals("task", "query")
    assert not s.kwarg_equals("missing_key", "anything")


@pytest.mark.basic
def test_has_flag_kwarg_form(monkeypatch):
    _argv(monkeypatch, "wdoc", "--completion", "--task=query")
    s = ArgvState()
    assert s.has_flag("completion")


@pytest.mark.basic
def test_has_flag_bare_argv_form(monkeypatch):
    # Bare --completion that fire didn't put into kwargs still counts.
    _argv(monkeypatch, "wdoc", "parse", "--", "--completion")
    s = ArgvState()
    assert s.has_flag("completion")


@pytest.mark.basic
def test_has_flag_absent(monkeypatch):
    _argv(monkeypatch, "wdoc", "query", "--path=/tmp/x")
    s = ArgvState()
    assert not s.has_flag("completion")


@pytest.mark.basic
def test_has_arg_positional_or_kwarg(monkeypatch):
    _argv(monkeypatch, "wdoc", "path", "--task=query")
    s = ArgvState()
    assert s.has_arg("path")  # positional
    assert s.has_arg("task")  # kwarg
    assert not s.has_arg("query")


@pytest.mark.basic
def test_argv_contains(monkeypatch):
    _argv(monkeypatch, "wdoc", "parse", "--", "--completion")
    s = ArgvState()
    assert s.argv_contains(" -- --completion")
    assert s.argv_contains(" parse ")
    assert not s.argv_contains("--nope")


# ---------- realistic end-to-end sequences ----------


@pytest.mark.basic
def test_web_block_sequence(monkeypatch):
    # Reproduces the rewrite chain cli_launcher applies for `wdoc web foo`:
    #   remove `web`, force task=query, force filetype=ddg, then duplicate
    #   the lone positional to both path and query.
    _argv(monkeypatch, "wdoc", "web", "what is rust?")
    s = ArgvState()
    s.remove_positional("web")
    s.set_kwarg("task", "query", force=True)
    s.set_kwarg("filetype", "ddg", force=True)
    temp = s.args[0]
    s.remove_positional(temp)
    s.set_kwarg("path", temp)
    s.set_kwarg("query", temp)

    assert s.args == []
    assert s.kwargs == {
        "task": "query",
        "filetype": "ddg",
        "path": "what is rust?",
        "query": "what is rust?",
    }
    for expected in (
        "--task=query",
        "--filetype=ddg",
        "--path=what is rust?",
        "--query=what is rust?",
    ):
        assert expected in sys.argv
    assert "web" not in sys.argv
