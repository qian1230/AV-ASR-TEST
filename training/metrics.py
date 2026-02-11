import torch
import Levenshtein
from typing import List, Dict, Tuple, Optional
from configs import config


class WERCalculator:
    """词错误率（WER）计算器"""

    def __init__(self, config=None, blank_id: int = 0):
        self.config = config or config
        self.blank_id = blank_id

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """计算单个样本的WER"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0

        distance = self._levenshtein_distance(ref_words, hyp_words)
        wer = distance / len(ref_words)

        return wer

    def _levenshtein_distance(self, ref_words: List[str],
                            hyp_words: List[str]) -> int:
        """计算词级别的Levenshtein距离"""
        ref_len = len(ref_words)
        hyp_len = len(hyp_words)

        if ref_len == 0:
            return hyp_len
        if hyp_len == 0:
            return ref_len

        if ref_words == hyp_words:
            return 0

        matrix = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

        for i in range(ref_len + 1):
            matrix[i][0] = i
        for j in range(hyp_len + 1):
            matrix[0][j] = j

        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost
                )

        return matrix[ref_len][hyp_len]

    def compute_batch_wer(self, references: List[str],
                         hypotheses: List[str]) -> Dict[str, float]:
        """计算批次的WER"""
        total_words = 0
        total_errors = 0

        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()

            ref_len = len(ref_words)
            hyp_len = len(hyp_words)

            if ref_len == 0:
                continue

            total_words += ref_len
            distance = self._levenshtein_distance(ref_words, hyp_words)
            total_errors += distance

        wer = total_errors / total_words if total_words > 0 else 0.0

        return {
            'wer': wer,
            'total_words': total_words,
            'total_errors': total_errors
        }

    def compute_from_decoded(self, targets: torch.Tensor,
                           decoded: List[str],
                           target_lengths: torch.Tensor,
                           text_processor) -> Dict[str, float]:
        """从解码结果计算WER"""
        references = []
        for i in range(len(decoded)):
            target = targets[i, :target_lengths[i]]
            ref_text = text_processor.decode(target)
            references.append(ref_text)

        return self.compute_batch_wer(references, decoded)


class CERCalculator:
    """字符错误率（CER）计算器"""

    def __init__(self, config=None, blank_id: int = 0):
        self.config = config or config
        self.blank_id = blank_id

    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """计算单个样本的CER"""
        ref_chars = list(reference.lower().replace(' ', ''))
        hyp_chars = list(hypothesis.lower().replace(' ', ''))

        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0

        distance = self._levenshtein_distance(ref_chars, hyp_chars)
        cer = distance / len(ref_chars)

        return cer

    def _levenshtein_distance(self, ref_chars: List[str],
                              hyp_chars: List[str]) -> int:
        """计算字符级别的Levenshtein距离"""
        ref_len = len(ref_chars)
        hyp_len = len(hyp_chars)

        if ref_len == 0:
            return hyp_len
        if hyp_len == 0:
            return ref_len

        if ref_chars == hyp_chars:
            return 0

        matrix = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

        for i in range(ref_len + 1):
            matrix[i][0] = i
        for j in range(hyp_len + 1):
            matrix[0][j] = j

        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j - 1] + cost
                )

        return matrix[ref_len][hyp_len]

    def compute_batch_cer(self, references: List[str],
                        hypotheses: List[str]) -> Dict[str, float]:
        """计算批次的CER"""
        total_chars = 0
        total_errors = 0

        for ref, hyp in zip(references, hypotheses):
            ref_chars = list(ref.lower().replace(' ', ''))
            hyp_chars = list(hyp.lower().replace(' ', ''))

            ref_len = len(ref_chars)
            if ref_len == 0:
                continue

            total_chars += ref_len
            distance = self._levenshtein_distance(ref_chars, hyp_chars)
            total_errors += distance

        cer = total_errors / total_chars if total_chars > 0 else 0.0

        return {
            'cer': cer,
            'total_chars': total_chars,
            'total_errors': total_errors
        }


class AccuracyCalculator:
    """准确率计算器"""

    def __init__(self, config=None):
        self.config = config or config
        self.correct = 0
        self.total = 0

    def update(self, predictions: List[str], references: List[str]):
        """更新统计"""
        for pred, ref in zip(predictions, references):
            self.total += 1
            if pred.lower().strip() == ref.lower().strip():
                self.correct += 1

    def compute(self) -> float:
        """计算准确率"""
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        """重置统计"""
        self.correct = 0
        self.total = 0


class RunningMetrics:
    """运行指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.total_samples = 0
        self.total_loss = 0.0
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.wer_calculator = WERCalculator()
        self.cer_calculator = CERCalculator()

    def update(self, loss: float, wer: float, cer: float, num_samples: int = 1):
        """更新指标"""
        self.total_samples += num_samples
        self.total_loss += loss * num_samples
        self.total_wer += wer * num_samples
        self.total_cer += cer * num_samples

    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        if self.total_samples == 0:
            return {
                'loss': 0.0,
                'wer': 0.0,
                'cer': 0.0
            }

        return {
            'loss': self.total_loss / self.total_samples,
            'wer': self.total_wer / self.total_samples,
            'cer': self.total_cer / self.total_samples
        }

    def get_summary(self) -> str:
        """获取指标摘要"""
        metrics = self.compute()
        return f"Loss: {metrics['loss']:.4f}, WER: {metrics['wer']*100:.2f}%, CER: {metrics['cer']*100:.2f}%"


class MetricsAggregator:
    """指标聚合器（支持分布式训练）"""

    def __init__(self):
        self.metrics = {
            'loss': [],
            'wer': [],
            'cer': [],
            'samples': 0
        }

    def update(self, loss: float, wer: float, cer: float, num_samples: int):
        """更新指标"""
        self.metrics['loss'].append(loss * num_samples)
        self.metrics['wer'].append(wer * num_samples)
        self.metrics['cer'].append(cer * num_samples)
        self.metrics['samples'] += num_samples

    def compute(self) -> Dict[str, float]:
        """计算聚合指标"""
        total_samples = self.metrics['samples']
        if total_samples == 0:
            return {'loss': 0.0, 'wer': 0.0, 'cer': 0.0}

        total_loss = sum(self.metrics['loss'])
        total_wer = sum(self.metrics['wer'])
        total_cer = sum(self.metrics['cer'])

        return {
            'loss': total_loss / total_samples,
            'wer': total_wer / total_samples,
            'cer': total_cer / total_samples
        }

    def sync_for_distributed(self):
        """分布式同步（占位）"""
        pass
