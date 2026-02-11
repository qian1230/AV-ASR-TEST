import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List
from pathlib import Path
from configs import config


class VideoPreprocessor(nn.Module):
    """视频预处理模块：提取唇部区域并预处理"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.fps = self.config.VIDEO_FPS
        self.size = self.config.VIDEO_SIZE
        self.grayscale = self.config.VIDEO_GRAYSCALE

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        前向传播：原始视频帧 -> 预处理后的张量

        Args:
            video_frames: 原始帧序列 [T, H, W] 或 [T, C, H, W]

        Returns:
            processed_frames: 预处理后的帧序列 [T, C, H, W]
                             归一化到[-1, 1]
        """
        if video_frames.dim() == 3:
            video_frames = video_frames.unsqueeze(1)

        if video_frames.dim() != 4:
            raise ValueError(f"Expected 3D or 4D video frames, got {video_frames.dim()}D")

        processed = []
        for frame in video_frames:
            if self.grayscale and frame.shape[0] == 3:
                frame = self._rgb_to_gray(frame)
            
            frame = frame.float() / 255.0
            
            if frame.shape[1] != self.size or frame.shape[2] != self.size:
                frame = F.interpolate(frame.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
            
            frame = frame * 2.0 - 1.0
            processed.append(frame)

        return torch.stack(processed, dim=0)

    def _rgb_to_gray(self, frame: torch.Tensor) -> torch.Tensor:
        """RGB转灰度"""
        if frame.shape[0] == 3:
            return frame.mean(dim=0, keepdim=True)
        return frame

    @staticmethod
    def extract_frames_from_video(video_path: str, target_fps: int = 30) -> torch.Tensor:
        """从视频文件提取帧序列"""
        try:
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if original_fps <= 0:
                return torch.zeros(0, 64, 64, dtype=torch.uint8)

            frame_indices = np.linspace(0, frame_count - 1,
                                         int(frame_count / original_fps * target_fps),
                                         dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (64, 64))
                    frames.append(torch.from_numpy(frame))

            cap.release()
            return torch.stack(frames) if frames else torch.zeros(0, 64, 64, dtype=torch.uint8)

        except Exception as e:
            print(f"Error loading video file {video_path}: {e}")
            return torch.zeros(0, 64, 64, dtype=torch.uint8)

    @staticmethod
    def extract_lip_region(frames: torch.Tensor,
                           lip_detector=None) -> torch.Tensor:
        """
        提取唇部区域

        Args:
            frames: 帧序列 [T, H, W, C]
            lip_detector: 唇部检测器（可选，默认使用简单规则）

        Returns:
            lip_frames: 唇部区域帧序列 [T, lip_h, lip_w, C]
        """
        if frames.numel() == 0:
            return frames

        lip_frames = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()

            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            lip_region = VideoPreprocessor._detect_simple_lip_region(frame_uint8)
            lip_frames.append(lip_region)

        return torch.stack([torch.from_numpy(f) for f in lip_frames])

    @staticmethod
    def _detect_simple_lip_region(image: np.ndarray) -> np.ndarray:
        """
        简单的唇部区域检测（基于人脸大致位置）

        在实际应用中，应使用更精确的检测器如：
        - dlib + shape_predictor
        - MediaPipe Face Mesh
        - OpenCV级联分类器
        """
        height, width = image.shape[:2]

        face_roi_y1 = int(height * 0.45)
        face_roi_y2 = int(height * 0.85)
        face_roi_x1 = int(width * 0.25)
        face_roi_x2 = int(width * 0.75)

        face_roi = image[face_roi_y1:face_roi_y2, face_roi_x1:face_roi_x2]

        if face_roi.size == 0:
            face_roi = cv2.resize(image, (64, 64))
            return face_roi

        lip_roi_y1 = int(face_roi.shape[0] * 0.55)
        lip_roi = face_roi[lip_roi_y1:, :]

        lip_resized = cv2.resize(lip_roi, (64, 64))

        if len(lip_resized.shape) == 2:
            lip_resized = cv2.cvtColor(lip_resized, cv2.COLOR_GRAY2RGB)

        return lip_resized


class SimpleVideoEncoderFeatureExtractor(nn.Module):
    """简单CNN视频特征提取器（用于端到端学习）"""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Linear(hidden_channels * 4, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入视频帧 [B, T, C, H, W]

        Returns:
            features: 视频特征 [B, T, D]
        """
        batch_size, time_steps, C, H, W = x.shape
        x = x.view(batch_size * time_steps, C, H, W)
        x = self.encoder(x)
        x = x.view(batch_size * time_steps, -1)
        x = self.projection(x)
        x = x.view(batch_size, time_steps, -1)
        return x


def temporal_downsample(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    """时序下采样到目标帧数"""
    current_frames = features.shape[1]
    if current_frames == target_frames:
        return features

    indices = torch.linspace(0, current_frames - 1, target_frames).long()
    indices = indices.clamp(max=current_frames - 1)

    return features[:, indices, :]
