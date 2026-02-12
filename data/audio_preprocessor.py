import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Optional, List
from configs import config


class AudioPreprocessor(nn.Module):
    """音频预处理模块：提取log-mel谱图特征"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.sample_rate = self.config.AUDIO_SAMPLE_RATE
        self.n_mels = self.config.AUDIO_N_MELS
        self.window_size = self.config.AUDIO_WINDOW_SIZE
        self.hop_length = self.config.AUDIO_HOP_LENGTH
        self.n_fft = int(self.sample_rate * self.window_size)

        self.mel_spec_transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=int(self.sample_rate * self.window_size),
                hop_length=int(self.sample_rate * self.hop_length),
                n_mels=self.n_mels,
                normalized=True,
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )

        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """预计算全局归一化参数"""
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        前向传播：音频波形 -> log-mel谱图

        Args:
            waveform: 音频波形 [B, T] 或 [T]

        Returns:
            log_mel_spec: log-mel谱图特征 [B, C, F, T]
                         C=1 (通道), F=n_mels, T=时间帧数
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() != 2:
            raise ValueError(f"Expected 1D or 2D waveform, got {waveform.dim()}D")

        mel_spec = self.mel_spec_transform(waveform)

        # 确保mel_spec不为0
        mel_spec = torch.clamp(mel_spec, min=1e-9)
        log_mel_spec = torch.log(mel_spec)

        return log_mel_spec

    @staticmethod
    def extract_features_from_file(audio_path: str) -> torch.Tensor:
        """从音频文件提取特征"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return torch.zeros(1, 16000)

    def get_frame_count(self, duration: float) -> int:
        """根据时长计算帧数"""
        return int(duration / self.hop_length)


class DynamicPaddingCollator:
    """动态padding批处理器"""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        audio_features = []
        video_features = []
        audio_lengths = []
        video_lengths = []
        targets = []
        target_lengths = []
        transcripts = []

        for item in batch:
            audio_features.append(item['audio_features'])
            video_features.append(item['video_features'])
            audio_lengths.append(item['audio_length'])
            video_lengths.append(item['video_length'])
            targets.append(item['targets'])
            target_lengths.append(item['target_length'])
            transcripts.append(item['transcript'])

        audio_features = self._pad_sequence(audio_features, dim=-1)
        video_features = self._pad_sequence(video_features, dim=0)

        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        video_lengths = torch.tensor(video_lengths, dtype=torch.long)
        targets = self._pad_sequence(targets, dim=-1)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        return {
            'audio_features': audio_features,
            'video_features': video_features,
            'audio_lengths': audio_lengths,
            'video_lengths': video_lengths,
            'targets': targets,
            'target_lengths': target_lengths,
            'transcript': transcripts
        }

    @staticmethod
    def _pad_sequence(sequences: List[torch.Tensor], dim: int = -1, pad_value: float = 0.0) -> torch.Tensor:
        """填充序列到批次最大长度"""
        if not sequences:
            return torch.tensor([])

        if dim == -1:
            # 假设所有序列都是 (C, T) 或 (C, F, T)，我们想在最后一个维度 T 上 pad
            max_len = max(s.shape[-1] for s in sequences)
            padded = []
            for seq in sequences:
                pad_len = max_len - seq.shape[-1]
                if pad_len > 0:
                     # F.pad 的 padding 参数是 (left, right, top, bottom, ...)
                     # 只需要 pad 最后一个维度
                    pad_config = [0] * (2 * seq.dim())
                    pad_config[0] = pad_len # pad last dim right side
                    padded.append(torch.nn.functional.pad(seq, pad_config, value=pad_value))
                else:
                    padded.append(seq)
            return torch.stack(padded, dim=0)
        elif dim == 0:
            # 针对视频序列 (T, C, H, W)，在第一个维度 T 上 pad
            max_len = max(s.shape[0] for s in sequences)
            padded = []
            for seq in sequences:
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    pad_config = [0] * (2 * seq.dim())
                    # pad dim 0 corresponds to the last pair in F.pad arguments (reversed order)
                    # For 4D input (T, C, H, W), dim 0 is padding index 7 (if 1-based) or index 6/7 in list
                    # F.pad expects: (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, ..., pad_first_dim_left, pad_first_dim_right)
                    # So for dim 0, we need to set the LAST two elements of pad_config
                    pad_config[-1] = pad_len
                    padded.append(torch.nn.functional.pad(seq, pad_config, value=pad_value))
                else:
                    padded.append(seq)
            return torch.stack(padded, dim=0)

        # Fallback for other dimensions (simplified)
        max_len = max(s.shape[dim] for s in sequences)
        padded = []
        for seq in sequences:
             # This generic logic is complex to get right for arbitrary dims with F.pad
             # For now, let's rely on the specific cases above which cover audio and video
             # Or implement a simple concat approach if needed
             pass
        return torch.stack(sequences, dim=0) # Placeholder if logic not hit


def compute_global_normalization(data_loader: torch.utils.data.DataLoader,
                                 preprocessor: AudioPreprocessor) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算全局归一化参数（均值和标准差）"""
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch in data_loader:
        audio_features = batch['audio_features']
        total_sum += audio_features.sum()
        total_sq_sum += (audio_features ** 2).sum()
        total_count += audio_features.numel()

    mean = total_sum / total_count
    std = torch.sqrt(total_sq_sum / total_count - mean ** 2)
    std = torch.clamp(std, min=1e-6)

    return mean, std
