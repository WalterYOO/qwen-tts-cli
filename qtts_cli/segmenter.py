"""Semantic sentence segmentation (FR-P0-04).

Invariant: ``"".join(segment_text(text)) == text`` — segmentation never drops or
rewrites a character, so subtitle cues built from segments reproduce the input
verbatim.
"""

from __future__ import annotations

_OPEN_TO_CLOSE = {
    "「": "」",
    "『": "』",
    "“": "”",
    "‘": "’",
    '"': '"',
    "'": "'",
    "(": ")",
    "（": "）",
    "[": "]",
    "【": "】",
    "《": "》",
    "<": ">",
}
_CLOSE_CHARS = set(_OPEN_TO_CLOSE.values())

# Chinese/Japanese sentence enders split unconditionally.
_CJK_ENDERS = set("。！？；…")
# ASCII enders that require lookahead to avoid decimals/abbreviations.
_ASCII_ENDERS = set(".!?;")

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "no",
    "fig",
    "inc",
    "ltd",
    "co",
    "approx",
}

_SECONDARY = set("，,、：:—–-")
_SPACES = set(" \t　")


def _prev_word(text: str, dot: int) -> str:
    j = dot - 1
    while j >= 0 and (text[j].isalnum() or text[j] in ".-"):
        j -= 1
    return text[j + 1 : dot].lower()


def _is_sentence_end(text: str, i: int) -> bool:
    ch = text[i]
    if ch in _CJK_ENDERS:
        return True
    if ch in _ASCII_ENDERS:
        if ch == ".":
            if i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
                return False
            if _prev_word(text, i) in _ABBREVIATIONS:
                return False
        j = i + 1
        while j < len(text) and text[j] in _ASCII_ENDERS:
            j += 1
        if j >= len(text):
            return True
        nxt = text[j]
        if nxt in _SPACES or nxt == "\n" or nxt in _CLOSE_CHARS:
            return True
        if nxt in _CJK_ENDERS or nxt in _OPEN_TO_CLOSE or "一" <= nxt <= "鿿":
            return True
        return False
    return False


def segment_text(text: str, *, max_chars: int | None = None) -> list[str]:
    """Split into sentence-like segments, preserving every character."""
    if not text:
        return []
    segments: list[str] = []
    start = 0
    i = 0
    stack: list[str] = []
    n = len(text)

    while i < n:
        ch = text[i]
        if ch in _OPEN_TO_CLOSE:
            # Plain apostrophes inside words (don't) must not open a quote.
            if ch == "'" and i > 0 and i + 1 < n and text[i - 1].isalnum() and text[i + 1].isalnum():
                i += 1
                continue
            stack.append(ch)
        elif stack and ch == _OPEN_TO_CLOSE[stack[-1]]:
            stack.pop()

        at_quote_top = not stack
        boundary = False

        if ch == "\n":
            boundary = True
        elif at_quote_top and _is_sentence_end(text, i):
            boundary = True
            j = i + 1
            while j < n and (text[j] in _CJK_ENDERS or text[j] in _ASCII_ENDERS or text[j] in _CLOSE_CHARS):
                j += 1
            i = j - 1

        if boundary:
            end = i + 1
            if ch in _CJK_ENDERS or ch in _ASCII_ENDERS:
                while end < n and text[end] in _SPACES:
                    end += 1
            piece = text[start:end]
            if piece:
                segments.append(piece)
            start = end
        i += 1

    if start < n:
        remainder = text[start:]
        if remainder:
            segments.append(remainder)

    if max_chars and max_chars > 0:
        expanded: list[str] = []
        for seg in segments:
            expanded.extend(split_long(seg, max_chars))
        segments = expanded
    return segments


def split_long(text: str, max_chars: int) -> list[str]:
    """Split an over-budget segment at the best boundary before max_chars."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = _best_cut(rest, max_chars)
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


def _quote_depths(text: str) -> list[int]:
    """Depth of the quote/parenthesis stack after each character position."""
    depths = [0] * (len(text) + 1)
    stack: list[str] = []
    for i, ch in enumerate(text):
        if ch in _OPEN_TO_CLOSE:
            if ch == "'" and i > 0 and i + 1 < len(text) and text[i - 1].isalnum() and text[i + 1].isalnum():
                depths[i + 1] = len(stack)
                continue
            stack.append(ch)
        elif stack and ch == _OPEN_TO_CLOSE[stack[-1]]:
            stack.pop()
        depths[i + 1] = len(stack)
    return depths


def _best_cut(text: str, limit: int) -> int:
    floor = max(1, limit // 2)
    upper = min(limit, len(text) - 1)
    depths = _quote_depths(text)
    # Prefer punctuation/whitespace that is not inside an open quote (P1-10).
    for i in range(upper - 1, floor - 1, -1):
        if text[i] in _SECONDARY and depths[i + 1] == 0:
            return i + 1
    for i in range(upper - 1, floor - 1, -1):
        if text[i] in _SPACES and depths[i + 1] == 0:
            return i + 1
    # Long quoted passages have no depth-0 boundary; pick the shallowest valid
    # one (punctuation before whitespace before a token boundary).
    best: tuple[tuple[int, int, int], int] | None = None
    for i in range(upper - 1, floor - 1, -1):
        cut = i + 1
        if _cut_inside_token(text, cut):
            continue
        rank = 0 if text[i] in _SECONDARY else (1 if text[i] in _SPACES else 2)
        # Minimize quote depth first, then prefer punctuation/space, then the
        # latest position (negated so max() expresses that preference).
        key = (-depths[cut], -rank, i)
        if best is None or key > best[0]:
            best = (key, cut)
    if best is not None:
        return best[1]
    cut = upper
    while cut > floor and _cut_inside_token(text, cut):
        cut -= 1
    return cut


def _cut_inside_token(text: str, cut: int) -> bool:
    if cut <= 0 or cut >= len(text):
        return False
    prev_c, next_c = text[cut - 1], text[cut]
    if prev_c.isdigit() and next_c.isdigit():
        return True
    if prev_c.isalnum() and next_c.isalnum():
        cjk = ("一" <= prev_c <= "鿿") and ("一" <= next_c <= "鿿")
        return not cjk
    return False


def sentence_count(text: str) -> int:
    return len([s for s in segment_text(text) if s.strip()])
