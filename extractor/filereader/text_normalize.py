"""
文本规范化模块。

职责：
- 清理 OCR/PDF 转写常见异常空格（如“运 用于” -> “运用于”）。
- 清理页眉页脚与孤立页码等版面噪声。
- 将圈号脚注标记（①~⑳）替换为可追踪引用占位符（如 [ref_4]）。

输出：
- 返回规范化后的文本。
- 返回规范化统计信息，便于在处理链路中做质量监控。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


_CJK_RANGE = r"\u4e00-\u9fff"
_CJK_RE = f"[{_CJK_RANGE}]"
_CJK_PUNCT_RE = r"[，。！？；：、“”‘’（）《》〈〉【】…]"

_SPACE_BETWEEN_CJK = re.compile(rf"(?<={_CJK_RE})\s+(?={_CJK_RE})")
_SPACE_BEFORE_CJK_PUNCT = re.compile(rf"\s+(?={_CJK_PUNCT_RE})")
_SPACE_AFTER_OPEN_PUNCT = re.compile(r"(?<=[（《〈【“‘])\s+")
_FULLWIDTH_SPACE = re.compile(r"\u3000")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_EMPTY_LINES = re.compile(r"\n{3,}")

_HEADER_FOOTER_LINE = re.compile(
    r"(?m)^\s*[\u4e00-\u9fffA-Za-z0-9\s]{0,25}20\d{2}\.\d+\s*[＿_]+\s*\d+\s*$"
)
_ISOLATED_PAGE_NUM = re.compile(r"(?m)^\s*[＿_—-]*\s*\d+\s*[＿_—-]*\s*$")

_CIRCLED_NUM_TO_INDEX = {chr(code): code - 9311 for code in range(9312, 9333)}
_CIRCLED_NUM_RE = re.compile("[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")


@dataclass
class NormalizeStats:
    abnormal_space_fixes: int = 0
    noise_lines_removed: int = 0
    footnote_markers_replaced: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def _replace_and_count(
    pattern: re.Pattern[str], text: str, repl: str
) -> Tuple[str, int]:
    new_text, count = pattern.subn(repl, text)
    return new_text, count


def _remove_noise(text: str) -> Tuple[str, int]:
    total_removed = 0
    text, removed = _replace_and_count(_HEADER_FOOTER_LINE, text, "")
    total_removed += removed
    text, removed = _replace_and_count(_ISOLATED_PAGE_NUM, text, "")
    total_removed += removed
    return text, total_removed


def _replace_footnotes(text: str) -> Tuple[str, int]:
    replaced = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal replaced
        marker = match.group(0)
        replaced += 1
        return f"[ref_{_CIRCLED_NUM_TO_INDEX[marker]}]"

    return _CIRCLED_NUM_RE.sub(_sub, text), replaced


def normalize_text(text: str) -> Tuple[str, Dict[str, int]]:
    """规范化文本并返回规范化统计。"""
    if not text:
        return "", NormalizeStats().to_dict()

    stats = NormalizeStats()
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized, count = _replace_and_count(_FULLWIDTH_SPACE, normalized, " ")
    stats.abnormal_space_fixes += count
    normalized, count = _replace_and_count(_SPACE_BETWEEN_CJK, normalized, "")
    stats.abnormal_space_fixes += count
    normalized, count = _replace_and_count(_SPACE_BEFORE_CJK_PUNCT, normalized, "")
    stats.abnormal_space_fixes += count
    normalized, count = _replace_and_count(_SPACE_AFTER_OPEN_PUNCT, normalized, "")
    stats.abnormal_space_fixes += count

    normalized, removed = _remove_noise(normalized)
    stats.noise_lines_removed += removed

    normalized, replaced = _replace_footnotes(normalized)
    stats.footnote_markers_replaced += replaced

    normalized, _ = _replace_and_count(_TRAILING_SPACE, normalized, "")
    normalized, _ = _replace_and_count(_MULTI_EMPTY_LINES, normalized, "\n\n")

    return normalized.strip(), stats.to_dict()
