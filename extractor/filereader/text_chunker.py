"""
中文文本分块策略（触发条件版）

参数说明：
- CHUNK_SIZE：目标块长度。
- OVERLAP：相邻块重叠长度。
- MAX_TEXT_LENGTH：单次安全分词最大长度。

触发流程：
1. 参数校验触发
   - 当 chunk_size <= overlap：抛出异常，阻止无效重叠配置。
   - 当 chunk_size <= 0 / overlap < 0 / max_text_length <= 0：抛出异常。

2. HanLP 加载触发
   - 当 use_hanlp=True：尝试加载 HanLP 分词模型。
   - 当模型加载失败：tokenizer 置为 None，后续自动降级字符切分。

3. 超长文本预处理触发
   - 当 len(text) <= MAX_TEXT_LENGTH：不预分段，直接进入分词与分块。
   - 当 len(text) > MAX_TEXT_LENGTH：按段落预分段。
   - 当段落仍超长：按句号边界拆分；若无句边界则按固定窗口强制切分。

4. 安全分词触发
   - 当 tokenizer 可用且 len(text) <= MAX_TEXT_LENGTH：调用 HanLP 分词。
   - 当 tokenizer 不可用：降级为 list(text) 字符切分。
   - 当 len(text) > MAX_TEXT_LENGTH：降级为 list(text) 字符切分。
   - 当分词发生异常：捕获异常并降级为 list(text)。

5. 分块与重叠触发
   - 默认按 CHUNK_SIZE 截取主窗口。
   - 当窗口后方存在句末标点：优先把块尾对齐句子边界（允许小幅超长）。
   - 下一块起点默认取 end_pos - OVERLAP。
   - 当可回退到上一个句子边界且位置合法：优先用句边界作为下一块起点。
   - 当起点未推进（潜在死循环）：强制 start_pos = end_pos。

6. 输出触发
   - token 块会拼接成字符串块（List[str]）输出。
   - process_files 输出 (filename, content, chunks) 列表。
   - get_text_stats 输出长度、段落数、是否预处理、预估分块数等统计信息。
"""

from __future__ import annotations

import os
import re
import importlib
from typing import List, Tuple


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


CHUNK_SIZE_DEFAULT = _env_int("CHUNK_SIZE", 500)
OVERLAP_DEFAULT = _env_int("OVERLAP", 100)
MAX_TEXT_LENGTH_DEFAULT = _env_int("MAX_TEXT_LENGTH", 120000)


class ChineseTextChunker:
    """中文文本分块器，将长文本分割成带有重叠的文本块。"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        overlap: int = OVERLAP_DEFAULT,
        max_text_length: int = MAX_TEXT_LENGTH_DEFAULT,
        use_hanlp: bool = True,
        force_hanlp: bool = False,
    ):
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if overlap < 0:
            raise ValueError("overlap不能小于0")
        if max_text_length <= 0:
            raise ValueError("max_text_length必须大于0")
        if force_hanlp and not use_hanlp:
            raise ValueError("force_hanlp=True 时必须同时设置 use_hanlp=True")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        self.use_hanlp = use_hanlp
        self.force_hanlp = force_hanlp
        self.tokenizer = self._load_tokenizer() if self.use_hanlp else None
        if self.force_hanlp and self.tokenizer is None:
            raise ImportError("HanLP 强制模式已开启，但未能加载 HanLP 模型")

    @staticmethod
    def _load_tokenizer():
        try:
            hanlp = importlib.import_module("hanlp")
            return hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        except Exception:
            return None

    def process_files(
        self, file_contents: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, List[str]]]:
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results

    def _preprocess_large_text(self, text: str) -> List[str]:
        if len(text) <= self.max_text_length:
            return [text]

        target_segment_size = min(
            self.max_text_length, max(10000, self.max_text_length // 2)
        )
        paragraphs = text.split("\n\n")
        if len(paragraphs) < 5:
            paragraphs = text.split("\n")

        processed_segments: List[str] = []
        current_segment = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > target_segment_size:
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                processed_segments.extend(
                    self._split_long_paragraph(para, target_segment_size)
                )
            else:
                if len(current_segment) + len(para) + 2 > target_segment_size:
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    current_segment = (
                        f"{current_segment}\n\n{para}" if current_segment else para
                    )

        if current_segment:
            processed_segments.append(current_segment)

        return processed_segments

    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        if len(text) <= max_size:
            return [text]

        sentences = re.split(r"([。！？.!?])", text)
        combined_sentences: List[str] = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)

        if not combined_sentences:
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        segments: List[str] = []
        current_segment = ""
        for sentence in combined_sentences:
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                segments.extend(
                    [
                        sentence[i : i + max_size]
                        for i in range(0, len(sentence), max_size)
                    ]
                )
            else:
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence

        if current_segment:
            segments.append(current_segment)

        return segments

    def _safe_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            if self.tokenizer is None:
                if self.force_hanlp:
                    raise RuntimeError("HanLP 强制模式下 tokenizer 不可用")
                return list(text)
            if len(text) > self.max_text_length:
                if self.force_hanlp:
                    raise RuntimeError("HanLP 强制模式下文本长度超过 max_text_length")
                return list(text)
            tokens = self.tokenizer(text)
            return tokens if tokens else []
        except Exception:
            if self.force_hanlp:
                raise
            return list(text)

    def chunk_text(self, text: str) -> List[str]:
        if not text or len(text) < self.chunk_size / 10:
            tokens = self._safe_tokenize(text)
            return ["".join(tokens)] if tokens else []

        text_segments = self._preprocess_large_text(text)
        all_chunks: List[str] = []
        for segment in text_segments:
            token_chunks = self._chunk_single_segment(segment)
            all_chunks.extend(["".join(tokens) for tokens in token_chunks if tokens])

        return all_chunks

    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        if not text:
            return []

        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []

        chunks: List[List[str]] = []
        start_pos = 0

        while start_pos < len(all_tokens):
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))

            if end_pos < len(all_tokens):
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:
                    end_pos = sentence_end

            chunk = all_tokens[start_pos:end_pos]
            if chunk:
                chunks.append(chunk)

            if end_pos >= len(all_tokens):
                break

            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(
                all_tokens, overlap_start
            )

            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start

            if start_pos >= end_pos:
                start_pos = end_pos

        return chunks

    @staticmethod
    def _is_sentence_end(token: str) -> bool:
        return token in ["。", "！", "？"]

    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)

    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0

    def get_text_stats(self, text: str) -> dict:
        stats = {
            "text_length": len(text),
            "needs_preprocessing": len(text) > self.max_text_length,
            "estimated_chunks": max(1, len(text) // self.chunk_size),
            "paragraphs": len(text.split("\n\n")),
            "lines": len(text.split("\n")),
        }

        if stats["needs_preprocessing"]:
            segments = self._preprocess_large_text(text)
            stats["preprocessed_segments"] = len(segments)
            stats["max_segment_length"] = (
                max(len(seg) for seg in segments) if segments else 0
            )

        return stats
