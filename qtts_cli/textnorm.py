"""Text normalization (FR-P0-04) and per-segment language detection.

The normalized text is the content source of truth: subtitles are rendered from
it verbatim, never from ASR output.
"""

from __future__ import annotations

import re
import unicodedata

# Bumped whenever normalization changes output for the same input. Recorded in
# the sidecar so a recipe can be reproduced with the matching rules (P1-4).
NORMALIZATION_VERSION = 2

# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_HANGUL = re.compile(r"[가-힯]")
_KANA = re.compile(r"[぀-ヿ]")
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def detect_language(text: str) -> str:
    """Segment-level language guess. Returns a model language name or 'Auto'."""
    if not text:
        return "Auto"
    if _HANGUL.search(text):
        return "Korean"
    if _KANA.search(text) and not _CJK.search(text):
        return "Japanese"
    if _CJK.search(text):
        return "Chinese"
    if _CYRILLIC.search(text):
        return "Russian"
    if _LATIN.search(text):
        return "English"
    return "Auto"


# --------------------------------------------------------------------------- #
# Number verbalization
# --------------------------------------------------------------------------- #
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千"]
_CN_BIG = ["", "万", "亿", "万亿"]


def int_to_chinese(n: int) -> str:
    """0..9999_9999_9999 -> Chinese reading (pragmatic, not linguistic-exact)."""
    if n == 0:
        return "零"
    if n < 0:
        return "负" + int_to_chinese(-n)
    groups: list[int] = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    parts: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx]
        if group == 0:
            if parts and not parts[-1].endswith("零"):
                parts.append("零")
            continue
        text = _four_digits_to_chinese(group)
        if parts and group < 1000 and not parts[-1].endswith("零"):
            text = "零" + text
        parts.append(text + _CN_BIG[idx])
    out = "".join(parts).rstrip("零")
    out = out.replace("零零", "零")
    if out.startswith("一十"):
        out = out[1:]
    return out or "零"


def _four_digits_to_chinese(group: int) -> str:
    out: list[str] = []
    zero_pending = False
    for pos in range(3, -1, -1):
        digit = (group // (10**pos)) % 10
        if digit == 0:
            zero_pending = bool(out)
            continue
        if zero_pending:
            out.append("零")
            zero_pending = False
        out.append(_CN_DIGITS[digit] + _CN_UNITS[pos])
    return "".join(out)


_EN_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_to_english(n: int) -> str:
    if n < 0:
        return "minus " + int_to_english(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return _EN_TENS[tens] + ("-" + _EN_ONES[rem] if rem else "")
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return _EN_ONES[hundreds] + " hundred" + (" " + int_to_english(rem) if rem else "")
    for scale, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= scale:
            head, rem = divmod(n, scale)
            return int_to_english(head) + " " + name + (" " + int_to_english(rem) if rem else "")
    return _EN_ONES[n]  # pragma: no cover


def _digits_to_words(text: str, language: str) -> str:
    nums = re.sub(r"\D", "", text)
    if language == "English":
        return " ".join(_EN_ONES[int(ch)] for ch in nums)
    return "".join(_CN_DIGITS[int(ch)] for ch in nums)


def _number_to_words(num: str, language: str) -> str:
    if "." in num:
        whole, frac = num.split(".", 1)
        whole_txt = _number_to_words(whole, language)
        if language == "English":
            return whole_txt + " point " + _digits_to_words(frac, language)
        return whole_txt + "点" + _digits_to_words(frac, language)
    value = int(num)
    if language == "Chinese":
        return int_to_chinese(value)
    return int_to_english(value)


# --------------------------------------------------------------------------- #
# Specific normalizations
# --------------------------------------------------------------------------- #
_MONTHS_EN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_CURRENCY_CN = {"¥": "元", "￥": "元", "$": "美元", "€": "欧元", "£": "英镑"}
_CURRENCY_EN = {"¥": "yuan", "￥": "yuan", "$": "dollars", "€": "euros", "£": "pounds"}

_UNITS = {
    "kg": ("千克", "kilograms"),
    "g": ("克", "grams"),
    "km": ("千米", "kilometers"),
    "m": ("米", "meters"),
    "cm": ("厘米", "centimeters"),
    "mm": ("毫米", "millimeters"),
    "ml": ("毫升", "milliliters"),
    "l": ("升", "liters"),
    "L": ("升", "liters"),
    "GB": ("吉字节", "gigabytes"),
    "MB": ("兆字节", "megabytes"),
    "KB": ("千字节", "kilobytes"),
    "Hz": ("赫兹", "hertz"),
    "kHz": ("千赫兹", "kilohertz"),
    "℃": ("摄氏度", "degrees Celsius"),
    "°C": ("摄氏度", "degrees Celsius"),
    "m²": ("平方米", "square meters"),
    "km²": ("平方千米", "square kilometers"),
}

_URL = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f000-\U0001f2ff"
    "\U0000fe0f"
    "\U0000200d"
    "]+"
)
# \b fails next to CJK because Python's \w includes CJK: "日期2026-11-02" would
# keep the raw digits. ASCII-digit lookarounds work in both scripts (P1-10).
_DATE_YMD = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CURRENCY = re.compile(r"([¥￥$€£])\s?(\d+(?:\.\d+)?)")
_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|km|cm|mm|ml|GB|MB|KB|kHz|Hz|m²|km²|℃|°C|[gLl])(?![A-Za-z0-9])"
)
# Python's \w includes CJK, so plain \b would refuse to verbalize "2026年".
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9.])")
_YEAR_CN = re.compile(r"(?<!\d)(\d{4})(?=年)")


def _normalize_for_language(text: str, language: str) -> str:
    cn = language == "Chinese"

    def date_repl(m: re.Match[str]) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return m.group(0)
        if cn:
            # Years are read digit by digit: 2026 -> 二零二六, not 二千零二十六.
            year_txt = _digits_to_words(m.group(1), "Chinese")
            return f"{year_txt}年{int_to_chinese(month)}月{int_to_chinese(day)}日"
        return f"{_MONTHS_EN[month - 1]} {int_to_english(day)}, {int_to_english(year)}"

    def time_repl(m: re.Match[str]) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        if cn:
            hour_txt = int_to_chinese(hour) + "点"
            return hour_txt if minute == 0 else f"{hour_txt}{int_to_chinese(minute)}分"
        return f"{int_to_english(hour)} {int_to_english(minute)}"

    def percent_repl(m: re.Match[str]) -> str:
        num = _number_to_words(m.group(1), language)
        return f"百分之{num}" if cn else f"{num} percent"

    def currency_repl(m: re.Match[str]) -> str:
        symbol, num = m.group(1), m.group(2)
        spoken = _number_to_words(num, language)
        if cn:
            return f"{spoken}{_CURRENCY_CN.get(symbol, '')}"
        return f"{spoken} {_CURRENCY_EN.get(symbol, '')}"

    def unit_repl(m: re.Match[str]) -> str:
        num, unit = m.group(1), m.group(2)
        cn_unit, en_unit = _UNITS.get(unit, (unit, unit))
        spoken = _number_to_words(num, language)
        return f"{spoken}{cn_unit}" if cn else f"{spoken} {en_unit}"

    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _DATE_YMD.sub(date_repl, text)
    if cn:
        text = _YEAR_CN.sub(lambda m: _digits_to_words(m.group(1), "Chinese"), text)
    text = _TIME.sub(time_repl, text)
    text = _PERCENT.sub(percent_repl, text)
    text = _CURRENCY.sub(currency_repl, text)
    text = _UNIT.sub(unit_repl, text)
    text = _NUMBER.sub(lambda m: _number_to_words(m.group(0), language), text)
    return text


_CJK_PUNCT = set("，。！？；：、（）【】《》“”‘’…—～·「」『』〈〉〔〕")


def _nfkc_preserving_cjk_punct(text: str) -> str:
    """NFKC per character, but keep CJK punctuation intact.

    A plain NFKC pass would turn '，。！？' into ',.!?', which changes Chinese
    prosody and the subtitle text users expect to see.
    """
    return "".join(ch if ch in _CJK_PUNCT else unicodedata.normalize("NFKC", ch) for ch in text)


def normalize_text(text: str, language: str = "auto") -> str:
    """Normalize unicode, whitespace, emoji, numbers, dates, money and units.

    Newlines are preserved as paragraph breaks; each line is normalized on its
    own so mixed-language paragraphs keep working.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _nfkc_preserving_cjk_punct(text)
    text = text.replace("​", "").replace("﻿", "")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t ]+", " ", line).strip()
        if not line:
            lines.append("")
            continue
        lang = detect_language(line) if language.lower() == "auto" else language
        if lang not in ("Chinese", "English"):
            lines.append(line)
            continue
        lines.append(_normalize_for_language(line, lang).strip())
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
