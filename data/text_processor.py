import torch
import re
from typing import List, Tuple, Optional
from configs import config


class TextProcessor:
    """文本处理器：文本编码与解码"""

    def __init__(self, config=None):
        self.config = config or config
        self.lowercase = self.config.LOWERCASE
        self.keep_spaces = self.config.KEEP_SPACES
        self.vocab = self.config.get_vocab()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.blank_id = self.config.BLANK_ID

        self.special_tokens = {
            '<blank>': self.blank_id,
            '<unk>': len(self.vocab)
        }

    def encode(self, text: str) -> torch.Tensor:
        """
        文本编码：将文本转换为token索引序列

        Args:
            text: 输入文本（小写英文）

        Returns:
            tokens: token索引序列 [T]
        """
        if self.lowercase:
            text = text.lower()

        text = self._preprocess_text(text)
        tokens = []
        for char in text:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.special_tokens['<unk>'])

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """
        文本解码：将token索引序列转换为文本

        Args:
            indices: token索引序列 [T] 或 [T, logits]

        Returns:
            text: 解码后的文本
        """
        if indices.dim() == 2:
            indices = indices.argmax(dim=-1)

        indices = indices.tolist()
        decoded_chars = []
        prev_idx = -1

        for idx in indices:
            if idx == self.blank_id:
                prev_idx = idx
                continue

            if idx == prev_idx:
                continue

            if idx in self.idx_to_char:
                decoded_chars.append(self.idx_to_char[idx])

            prev_idx = idx

        text = ''.join(decoded_chars)
        text = self._postprocess_text(text)

        return text

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        allowed_chars = r'abcdefghijklmnopqrstuvwxyz !,.\?'

        filtered_chars = []
        for char in text:
            if char.lower() in allowed_chars:
                filtered_chars.append(char.lower() if self.lowercase else char)
            elif char == ' ' and self.keep_spaces:
                filtered_chars.append(' ')

        return ''.join(filtered_chars)

    def _postprocess_text(self, text: str) -> str:
        """文本后处理"""
        text = text.strip()
        return text

    def encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量文本编码

        Args:
            texts: 文本列表

        Returns:
            tokens: token序列 [B, T]
            lengths: 每个文本的长度 [B]
        """
        tokens_list = [self.encode(text) for text in texts]
        lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)

        max_len = lengths.max().item()
        padded_tokens = torch.zeros(len(tokens_list), max_len, dtype=torch.long)

        for i, token in enumerate(tokens_list):
            padded_tokens[i, :len(token)] = token

        return padded_tokens, lengths

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab) + len(self.special_tokens)

    def collapse_repeated_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """
        合并相邻重复token（CTC后处理）

        Args:
            indices: token索引序列 [T]

        Returns:
            collapsed: 合并后的序列 [T']
        """
        if indices.numel() == 0:
            return indices

        collapsed = []
        prev = -1
        for idx in indices.tolist():
            if idx != prev and idx != self.blank_id:
                collapsed.append(idx)
            prev = idx

        return torch.tensor(collapsed, dtype=torch.long)


class CharacterVocab:
    """字符级词汇表辅助类"""

    ENGLISH_CHARS = 'abcdefghijklmnopqrstuvwxyz'
    PUNCTUATION = r' !,.\?'
    DIGITS = '0123456789'

    @classmethod
    def get_default_vocab(cls, include_digits: bool = False) -> List[str]:
        """获取默认词汇表"""
        vocab = ['<blank>']
        vocab.extend(cls.PUNCTUATION)
        vocab.extend(cls.ENGLISH_CHARS)
        if include_digits:
            vocab.extend(cls.DIGITS)
        return vocab

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """计算字符错误率（CER）"""
        ref_chars = list(reference.replace(' ', ''))
        hyp_chars = list(hypothesis.replace(' ', ''))

        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0

        dp_matrix = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

        for i in range(len(ref_chars) + 1):
            dp_matrix[i][0] = i
        for j in range(len(hyp_chars) + 1):
            dp_matrix[0][j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    dp_matrix[i][j] = dp_matrix[i - 1][j - 1]
                else:
                    dp_matrix[i][j] = min(
                        dp_matrix[i - 1][j] + 1,
                        dp_matrix[i][j - 1] + 1,
                        dp_matrix[i - 1][j - 1] + 1
                    )

        return dp_matrix[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """计算词错误率（WER）"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0

        dp_matrix = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            dp_matrix[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp_matrix[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp_matrix[i][j] = dp_matrix[i - 1][j - 1]
                else:
                    dp_matrix[i][j] = min(
                        dp_matrix[i - 1][j] + 1,
                        dp_matrix[i][j - 1] + 1,
                        dp_matrix[i - 1][j - 1] + 1
                    )

        return dp_matrix[len(ref_words)][len(hyp_words)] / len(ref_words)
