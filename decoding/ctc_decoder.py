import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from configs import config


class CTCDecoder:
    """CTC解码器

    支持：
    - 贪心解码（Greedy Decoding）
    - 束搜索解码（Beam Search Decoding）
    """

    def __init__(self, config=None, blank_id: int = 0):
        self.config = config or config
        self.blank_id = blank_id

    def greedy_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """
        贪心解码：逐帧取概率最大的token

        Args:
            logits: 模型输出的logits [T, B, V] 或 [B, T, V]

        Returns:
            predictions: 解码后的token序列 [B, T']
        """
        if logits.dim() == 3:
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
            else:
                logits = logits.transpose(0, 1)

        log_probs = torch.log_softmax(logits, dim=-1)
        predictions = log_probs.argmax(dim=-1)

        predictions = self._collapse_repeated(predictions)

        return predictions

    def _collapse_repeated(self, predictions: torch.Tensor) -> torch.Tensor:
        """合并相邻重复token"""
        if predictions.numel() == 0:
            return predictions

        result = []
        prev = -1
        for idx in predictions.tolist():
            if idx != prev and idx != self.blank_id:
                result.append(idx)
            prev = idx

        return torch.tensor(result, dtype=torch.long)

    def beam_search_decode(self, logits: torch.Tensor,
                          beam_size: int = 5,
                          max_output_length: int = 100) -> List[torch.Tensor]:
        """
        束搜索解码

        Args:
            logits: 模型输出的logits [T, B, V] 或 [B, T, V]
            beam_size: 束大小
            max_output_length: 最大输出长度

        Returns:
            hypotheses: 解码假设列表
        """
        if logits.dim() == 3:
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
            else:
                logits = logits.transpose(0, 1)

        log_probs = torch.log_softmax(logits, dim=-1)

        hypotheses = []
        for b in range(log_probs.size(0)):
            hyp = self._beam_search_single(
                log_probs[b],
                beam_size,
                max_output_length
            )
            hypotheses.append(hyp)

        return hypotheses

    def _beam_search_single(self, log_probs: torch.Tensor,
                          beam_size: int,
                          max_output_length: int) -> torch.Tensor:
        """单条样本的束搜索"""
        T, V = log_probs.shape

        beam = [{'seq': [], 'score': 0.0, 'last_token': -1}]

        for t in range(T):
            new_beam = []
            for candidate in beam:
                last_token = candidate['last_token']
                for v in range(V):
                    if v == self.blank_id:
                        new_seq = candidate['seq']
                        new_score = candidate['score'] + log_probs[t, v].item()
                        new_last = self.blank_id
                    else:
                        if v == last_token:
                            new_seq = candidate['seq']
                        else:
                            new_seq = candidate['seq'] + [v]
                        new_score = candidate['score'] + log_probs[t, v].item()
                        new_last = v

                    new_candidate = {
                        'seq': new_seq,
                        'score': new_score,
                        'last_token': new_last
                    }
                    new_beam.append(new_candidate)

            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam = new_beam[:beam_size]

        best_hypothesis = beam[0]
        return torch.tensor(best_hypothesis['seq'], dtype=torch.long)

    def decode_with_lm(self, logits: torch.Tensor,
                      lm_model=None,
                      lm_weight: float = 0.5) -> torch.Tensor:
        """带语言模型的解码（实验性）"""
        predictions = self.greedy_decode(logits)
        return predictions

    def batch_decode(self, logits: torch.Tensor,
                    method: str = "greedy",
                    beam_size: int = 5) -> List[str]:
        """
        批量解码

        Args:
            logits: 模型输出的logits [B, T, V]
            method: 解码方法 ("greedy" 或 "beam")
            beam_size: 束大小

        Returns:
            hypotheses: 解码后的文本列表
        """
        if method == "greedy":
            predictions = self.greedy_decode(logits)
            hypotheses = []
            for pred in predictions:
                hyp = self._tokens_to_string(pred)
                hypotheses.append(hyp)
            return hypotheses

        elif method == "beam":
            predictions = self.beam_search_decode(logits, beam_size)
            hypotheses = []
            for pred in predictions:
                hyp = self._tokens_to_string(pred)
                hypotheses.append(hyp)
            return hypotheses

        else:
            raise ValueError(f"Unknown decoding method: {method}")

    def _tokens_to_string(self, tokens: torch.Tensor) -> str:
        """将token序列转换为字符串"""
        text = ""
        for idx in tokens.tolist():
            if idx == self.blank_id:
                continue
            if hasattr(self, 'idx_to_char') and idx in self.idx_to_char:
                text += self.idx_to_char[idx]
            else:
                text += str(idx) + " "

        return text.strip()


class BeamSearchDecoder:
    """优化的束搜索解码器"""

    def __init__(self, config=None, blank_id: int = 0, beam_size: int = 5):
        self.config = config or config
        self.blank_id = blank_id
        self.beam_size = beam_size

    def decode(self, log_probs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        执行束搜索解码

        Args:
            log_probs: 对数概率分布 [T, V]

        Returns:
            best_seq: 最佳序列
            best_score: 最佳分数
        """
        T, V = log_probs.shape

        beam_hyps = [{'seq': [], 'score': 0.0, 'last_token': -1, 'blank_runs': 0}]

        for t in range(T):
            new_beam = []

            for hyp in beam_hyps:
                for v in range(V):
                    score = log_probs[t, v].item()

                    if v == self.blank_id:
                        new_seq = hyp['seq']
                        new_score = hyp['score'] + score
                        new_last = -1
                        new_blank_runs = hyp['blank_runs'] + 1
                    else:
                        if hyp['last_token'] == v:
                            new_seq = hyp['seq'][:-1] + [v]
                        else:
                            new_seq = hyp['seq'] + [v]
                        new_score = hyp['score'] + score
                        new_last = v
                        new_blank_runs = 0

                    new_hyp = {
                        'seq': new_seq,
                        'score': new_score,
                        'last_token': new_last,
                        'blank_runs': new_blank_runs
                    }
                    new_beam.append(new_hyp)

            new_beam.sort(key=lambda x: x['score'], reverse=True)
            beam_hyps = new_beam[:self.beam_size]

        best_hyp = beam_hyps[0]
        return torch.tensor(best_hyp['seq']), best_hyp['score']


class PrefixBeamSearchDecoder:
    """前缀束搜索解码器（优化版）"""

    def __init__(self, config=None, blank_id: int = 0, beam_size: int = 5):
        self.config = config or config
        self.blank_id = blank_id
        self.beam_size = beam_size

    def decode(self, log_probs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """前缀束搜索解码实现"""
        T, V = log_probs.shape

        prefixes = {((), 0): 0.0}

        for t in range(T):
            new_prefixes = {}

            for (prefix, last_token), score in prefixes.items():
                for v in range(V):
                    new_score = score + log_probs[t, v].item()

                    if v == self.blank_id:
                        new_prefix = (prefix, last_token)
                        new_prefixes[new_prefix] = max(
                            new_prefixes.get(new_prefix, -float('inf')),
                            new_score
                        )
                    else:
                        if last_token == v:
                            new_prefix = (prefix, -1)
                        else:
                            new_prefix = (prefix + (v,), v)
                        new_prefixes[new_prefix] = max(
                            new_prefixes.get(new_prefix, -float('inf')),
                            new_score
                        )

            sorted_prefixes = sorted(
                new_prefixes.items(),
                key=lambda x: x[1],
                reverse=True
            )
            prefixes = dict(sorted_prefixes[:self.beam_size])

        best_prefix = max(prefixes.items(), key=lambda x: x[1])
        return torch.tensor(best_prefix[0][0]), best_prefix[1]


def create_decoder(config, decoder_type: str = "greedy", beam_size: int = 5) -> CTCDecoder:
    """工厂函数：创建CTC解码器"""
    if decoder_type == "greedy":
        return CTCDecoder(config)
    elif decoder_type == "beam":
        return CTCDecoder(config)
    elif decoder_type == "prefix":
        return PrefixBeamSearchDecoder(config, beam_size=beam_size)
    else:
        return CTCDecoder(config)
