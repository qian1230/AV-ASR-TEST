import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from configs import config
from .audio_preprocessor import AudioPreprocessor, DynamicPaddingCollator
from .video_preprocessor import VideoPreprocessor
from .text_processor import TextProcessor


class AVASRDataset(Dataset):
    """AV-ASR数据集类"""

    def __init__(self,
                 manifest_path: str,
                 audio_dir: str = None,
                 video_dir: str = None,
                 config=None,
                 transform=None,
                 is_training: bool = True):
        """
        Args:
            manifest_path: 数据清单文件路径（每行: audio_path video_path transcript）
            audio_dir: 音频文件目录
            video_dir: 视频文件目录
            config: 配置对象
            transform: 数据增强变换
            is_training: 是否为训练模式
        """
        self.config = config or config
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.video_dir = Path(video_dir) if video_dir else None
        self.is_training = is_training
        self.transform = transform

        self.audio_preprocessor = AudioPreprocessor(self.config)
        self.video_preprocessor = VideoPreprocessor(self.config)
        self.text_processor = TextProcessor(self.config)
        self.collator = DynamicPaddingCollator(pad_value=0.0)
        self.collate_fn = self.collator

        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """加载数据清单"""
        samples = []

        if not os.path.exists(manifest_path):
            return self._create_dummy_samples(100)

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')
                if len(parts) >= 3:
                    audio_path = parts[0].strip()
                    video_path = parts[1].strip()
                    transcript = parts[2].strip()

                    if self.audio_dir:
                        audio_path = str(self.audio_dir / audio_path)
                    if self.video_dir:
                        video_path = str(self.video_dir / video_path)

                    samples.append({
                        'audio_path': audio_path,
                        'video_path': video_path,
                        'transcript': transcript
                    })

        if len(samples) == 0:
            samples = self._create_dummy_samples(100)

        return samples

    def _create_dummy_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """创建虚拟样本用于测试"""
        dummy_transcripts = [
            "hello world",
            "this is a test",
            "speech recognition is fun",
            "audio and video together",
            "machine learning rocks",
            "good morning everyone",
            "how are you today",
            "thank you very much",
            "welcome to our presentation",
            "let us begin now"
        ]

        samples = []
        for i in range(num_samples):
            transcript = random.choice(dummy_transcripts)
            samples.append({
                'audio_path': f"dummy_audio_{i}.wav",
                'video_path': f"dummy_video_{i}.mp4",
                'transcript': transcript
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        video_path = sample['video_path']
        transcript = sample['transcript']

        audio_features = self._load_and_process_audio(audio_path)
        video_features = self._load_and_process_video(video_path)
        targets = self.text_processor.encode(transcript)

        audio_length = torch.tensor(audio_features.shape[-1], dtype=torch.long)
        video_length = torch.tensor(video_features.shape[0], dtype=torch.long)
        target_length = torch.tensor(len(targets), dtype=torch.long)
        targets_tensor = targets.clone().detach()

        return {
            'audio_features': audio_features,
            'video_features': video_features,
            'audio_length': audio_length,
            'video_length': video_length,
            'targets': targets_tensor,
            'target_length': target_length,
            'transcript': transcript
        }

    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """加载并处理音频"""
        try:
            if audio_path.startswith('dummy'):
                max_frames = 500
                return torch.randn(1, self.config.AUDIO_N_MELS, max_frames)

            waveform = self.audio_preprocessor.extract_features_from_file(audio_path)
            audio_features = self.audio_preprocessor(waveform)

            if self.transform and random.random() < 0.3:
                audio_features = self._apply_audio_augmentation(audio_features)

            return audio_features

        except Exception as e:
            return torch.randn(1, self.config.AUDIO_N_MELS, 500)

    def _load_and_process_video(self, video_path: str) -> torch.Tensor:
        """加载并处理视频"""
        try:
            if video_path.startswith('dummy'):
                max_frames = 150
                if self.config.VIDEO_GRAYSCALE:
                    return torch.randn(max_frames, 1, 64, 64)
                return torch.randn(max_frames, 3, 64, 64)

            frames = self.video_preprocessor.extract_frames_from_video(video_path)

            if frames.numel() == 0:
                return torch.randn(150, 1, 64, 64)

            frames = frames.float() / 255.0
            if self.config.VIDEO_GRAYSCALE and frames.shape[-1] == 3:
                frames = frames.mean(dim=-1, keepdim=True)

            # Permute to (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2)

            if self.transform and random.random() < 0.3:
                frames = self._apply_video_augmentation(frames)

            return frames

        except Exception as e:
            return torch.randn(150, 1, 64, 64)

    def _apply_audio_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """音频数据增强"""
        if random.random() < 0.5:
            speed_factor = random.uniform(0.9, 1.1)
            audio = self._speed_augment(audio, speed_factor)

        if random.random() < 0.5:
            audio = audio + torch.randn_like(audio) * 0.01

        return audio

    def _apply_video_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """视频数据增强"""
        if random.random() < 0.3:
            brightness = random.uniform(0.9, 1.1)
            video = video * brightness

        if random.random() < 0.3:
            flip = random.random() < 0.5
            if flip:
                video = torch.flip(video, dims=[-1])

        return video

    def _speed_augment(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        """速度增强"""
        orig_length = audio.shape[-1]
        new_length = int(orig_length / factor)
        indices = torch.linspace(0, orig_length - 1, new_length).long()
        indices = indices.clamp(max=orig_length - 1)
        audio = torch.index_select(audio, -1, indices)
        return audio


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证和测试数据加载器"""
    data_dir = Path(config.DATA_DIR)

    train_manifest = data_dir / "train_manifest.txt"
    val_manifest = data_dir / "val_manifest.txt"
    test_manifest = data_dir / "test_manifest.txt"

    train_dataset = AVASRDataset(
        manifest_path=str(train_manifest) if train_manifest.exists() else "",
        audio_dir=str(data_dir / "audio"),
        video_dir=str(data_dir / "video"),
        config=config,
        is_training=True
    )

    val_dataset = AVASRDataset(
        manifest_path=str(val_manifest) if val_manifest.exists() else "",
        audio_dir=str(data_dir / "audio"),
        video_dir=str(data_dir / "video"),
        config=config,
        is_training=False
    )

    test_dataset = AVASRDataset(
        manifest_path=str(test_manifest) if test_manifest.exists() else "",
        audio_dir=str(data_dir / "audio"),
        video_dir=str(data_dir / "video"),
        config=config,
        is_training=False
    )

    collator = DynamicPaddingCollator(pad_value=0.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
