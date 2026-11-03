"""Text-side QC helpers for ASR comparison (FR-P1-03).

Pure Python: importing this module never imports torch/qwen_asr, so
normalisation and edit-distance rules are unit-testable without models.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Optional, Sequence

from .catalog import display_language
from .errors import UsageError
from .textnorm import detect_language, normalize_text

METRICS = ("auto", "cer", "wer")
_CJK_METRIC_LANGUAGES = {"chinese", "japanese", "cantonese"}
_CJK_SCRIPT = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_KANA_SCRIPT = re.compile(r"[぀-ヿ]")
_LATIN_SCRIPT = re.compile(r"[A-Za-z]")


def qc_language(language: Optional[str], text: str = "") -> str:
    """Resolve one language for the whole QC comparison.

    ``auto`` must be resolved once from the reference text and then reused for
    the ASR transcript; detecting each side separately would verbalize digits
    differently (t10 H-B02).
    """
    value = (language or "auto").strip()
    if value and value.lower() != "auto":
        # Canonicalise before normalize_text(): lower-case "chinese" must still
        # expand digits/dates/currency exactly like "Chinese" (t10 P1-2).
        return display_language(value)
    cjk = len(_CJK_SCRIPT.findall(text or ""))
    kana = len(_KANA_SCRIPT.findall(text or ""))
    latin = len(_LATIN_SCRIPT.findall(text or ""))
    if kana and kana + cjk > latin:
        return "Japanese"
    if cjk > latin:
        return "Chinese"
    if latin:
        return "English"
    fallback = detect_language(text or "")
    return "English" if fallback == "Auto" else fallback


def resolve_metric(
    language: Optional[str],
    text: str = "",
    metric: str = "auto",
) -> str:
    """Resolve ``auto`` to ``cer`` for CJK-dominant text, else ``wer``."""
    value = (metric or "auto").strip().lower()
    if value not in METRICS:
        raise UsageError(
            f"unsupported metric: {metric!r}; expected auto, cer or wer",
            details={"metric": metric, "supported": list(METRICS)},
        )
    if value != "auto":
        return value
    return "cer" if qc_language(language, text).lower() in _CJK_METRIC_LANGUAGES else "wer"


def normalize_for_qc(text: str, language: str = "auto") -> str:
    """Normalize text for CER/WER: NFKC/numbers first, then drop punctuation.

    Both the reference and the ASR hypothesis go through this exact pipeline so
    differences come from recognition, not formatting.
    """
    if not text:
        return ""
    resolved = qc_language(language, str(text))
    normalized = normalize_text(str(text), resolved)
    stripped = "".join(
        ch for ch in normalized if unicodedata.category(ch)[0] not in ("P", "S")
    )
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return collapsed.casefold()


# Freeze the review's name as an alias; callers may use either spelling.
qc_normalize = normalize_for_qc


def text_units(text: str, metric: str = "cer") -> list[str]:
    """Split normalized text into comparison units.

    CER compares every non-whitespace character; WER compares whitespace words.
    """
    if metric == "cer":
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Classic Levenshtein distance over arbitrary comparable sequences."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_unit in enumerate(reference, 1):
        current = [i]
        for j, hyp_unit in enumerate(hypothesis, 1):
            cost = 0 if ref_unit == hyp_unit else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion from reference
                    current[j - 1] + 1,  # insertion into hypothesis
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def error_ratio(distance: int, reference_units: Sequence[Any]) -> float:
    return float(distance) / float(max(1, len(reference_units)))


def _get(obj: Any, *names: str) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _item_spans(
    items: Optional[Sequence[Any]],
    *,
    separator: str,
    language: str,
) -> list[tuple[int, int, Optional[float], Optional[float]]]:
    """Map each timestamp item to a range of hypothesis units."""
    spans: list[tuple[int, int, Optional[float], Optional[float]]] = []
    if not items:
        return spans
    offset = 0
    for item in items:
        raw = _get(item, "text", "word") or ""
        normalized = normalize_for_qc(str(raw), language)
        count = len(normalized.split()) if separator else len(
            [ch for ch in normalized if not ch.isspace()]
        )
        if count:
            spans.append(
                (
                    offset,
                    offset + count,
                    _number(_get(item, "start", "start_time")),
                    _number(_get(item, "end", "end_time")),
                )
            )
        offset += count
    return spans


def _span_time(
    spans: Sequence[tuple[int, int, Optional[float], Optional[float]]],
    start: int,
    end: int,
) -> tuple[Optional[float], Optional[float]]:
    hits = [span for span in spans if span[0] < end and span[1] > start]
    if not hits:
        return None, None
    starts = [hit[2] for hit in hits if hit[2] is not None]
    ends = [hit[3] for hit in hits if hit[3] is not None]
    return (
        min(starts) if starts else None,
        max(ends) if ends else None,
    )


def error_blocks(
    reference_units: Sequence[str],
    hypothesis_units: Sequence[str],
    *,
    items: Optional[Sequence[Any]] = None,
    separator: str = "",
    language: str = "auto",
) -> list[dict[str, Any]]:
    """Describe replace/delete/insert blocks with spans and optional times.

    ``items`` are ASR timestamp items (``text/start/end``); when supplied,
    blocks on the hypothesis side carry the matching ``start``/``end``.
    A deletion has no hypothesis span, so its times stay ``None``.
    """
    matcher = SequenceMatcher(
        a=list(reference_units), b=list(hypothesis_units), autojunk=False
    )
    spans = _item_spans(items, separator=separator, language=language)
    blocks: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        start: Optional[float] = None
        end: Optional[float] = None
        if op in ("replace", "insert") and j1 < j2:
            start, end = _span_time(spans, j1, j2)
        blocks.append(
            {
                "op": op,
                "ref": separator.join(reference_units[i1:i2]),
                "hyp": separator.join(hypothesis_units[j1:j2]),
                "ref_span": [i1, i2],
                "hyp_span": [j1, j2],
                "start": start,
                "end": end,
            }
        )
    return blocks
