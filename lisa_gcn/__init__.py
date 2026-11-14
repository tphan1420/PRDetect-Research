"""LISA-GCN Core: a standalone, source-only extension of PRDetect GCN."""

from .windowing import SentenceSpan, Window, plan_sentence_windows

__all__ = ("SentenceSpan", "Window", "plan_sentence_windows")
