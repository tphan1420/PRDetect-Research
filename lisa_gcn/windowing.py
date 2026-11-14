"""Sentence-aligned partitions used by LISA-GCN.

The planner is deliberately independent of spaCy/PyTorch, so its coverage
invariants can be tested without downloading an encoder.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError("a sentence span must be a non-empty [start, end) interval")


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    split_sentence: bool = False


def plan_sentence_windows(token_count: int, sentences: Sequence[SentenceSpan], window_size: int) -> list[Window]:
    """Partition every token exactly once, preferring complete sentences.

    A sentence longer than ``window_size`` is split into consecutive fixed-size
    pieces and marked, making the exception observable in cache metadata.
    """
    if token_count < 0 or window_size < 1:
        raise ValueError("token_count must be non-negative and window_size positive")
    if token_count == 0:
        return []
    if not sentences or sentences[0].start != 0 or sentences[-1].end != token_count:
        raise ValueError("sentence spans must cover the complete token sequence")
    if any(a.end != b.start for a, b in zip(sentences, sentences[1:])):
        raise ValueError("sentence spans must be contiguous and non-overlapping")

    result: list[Window] = []
    pending_start = pending_end = None
    for sentence in sentences:
        length = sentence.end - sentence.start
        if length > window_size:
            if pending_start is not None:
                result.append(Window(pending_start, pending_end))
                pending_start = pending_end = None
            for start in range(sentence.start, sentence.end, window_size):
                result.append(Window(start, min(start + window_size, sentence.end), split_sentence=True))
            continue
        if pending_start is None:
            pending_start, pending_end = sentence.start, sentence.end
        elif sentence.end - pending_start <= window_size:
            pending_end = sentence.end
        else:
            result.append(Window(pending_start, pending_end))
            pending_start, pending_end = sentence.start, sentence.end
    if pending_start is not None:
        result.append(Window(pending_start, pending_end))

    covered = [token for window in result for token in range(window.start, window.end)]
    if covered != list(range(token_count)):
        raise AssertionError("window planner violated its exact-coverage contract")
    return result


def sentence_spans_from_doc(doc: Iterable[object]) -> list[SentenceSpan]:
    """Read verified spans from a spaCy ``Doc`` without guessing token order."""
    return [SentenceSpan(sentence.start, sentence.end) for sentence in doc.sents]
