import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class FeatureFusion(nn.Module):
    """特征融合模块

    支持多种融合策略：
    - simple: 简单拼接
    - weighted: 加权拼接（可学习权重）
    - attention: 注意力融合
    """

    def __init__(self, config=None, fusion_type: str = "simple"):
        super().__init__()
        self.config = config or config
        self.audio_dim = self.config.AUDIO_ENCODER_FEATURE_DIM
        self.video_dim = self.config.VIDEO_ENCODER_FEATURE_DIM
        self.fusion_type = fusion_type

        if fusion_type == "simple":
            self.fused_dim = self.audio_dim + self.video_dim
            self.fusion_layer = nn.Identity()

        elif fusion_type == "weighted":
            self.fused_dim = self.audio_dim + self.video_dim
            self.audio_weight = nn.Parameter(torch.tensor(0.5))
            self.video_weight = nn.Parameter(torch.tensor(0.5))
            self.fusion_layer = nn.Identity()

        elif fusion_type == "attention":
            self.fused_dim = self.audio_dim
            self.audio_attention = nn.Sequential(
                nn.Linear(self.audio_dim, self.audio_dim // 4),
                nn.Tanh(),
                nn.Linear(self.audio_dim // 4, 1)
            )
            self.video_attention = nn.Sequential(
                nn.Linear(self.video_dim, self.video_dim // 4),
                nn.Tanh(),
                nn.Linear(self.video_dim // 4, 1)
            )
            self.fusion_layer = nn.Identity()

        elif fusion_type == "bilinear":
            self.fused_dim = self.audio_dim
            self.bilinear = nn.Bilinear(self.audio_dim, self.video_dim, self.fused_dim)
            self.fusion_layer = nn.Identity()

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Bilinear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, audio_features: torch.Tensor,
               video_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: 音频特征 [B, T_a, D_a]
            video_features: 视频特征 [B, T_v, D_v]

        Returns:
            fused_features: 融合特征 [B, T, D_fused]
        """
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1)

        B_a, T_a, D_a = audio_features.shape
        B_v, T_v, D_v = video_features.shape

        assert B_a == B_v, f"Batch size mismatch: {B_a} vs {B_v}"

        T = min(T_a, T_v)

        audio_features = audio_features[:, :T, :]
        video_features = video_features[:, :T, :]

        if self.fusion_type == "simple":
            fused = torch.cat([audio_features, video_features], dim=-1)

        elif self.fusion_type == "weighted":
            audio_weight = torch.sigmoid(self.audio_weight)
            video_weight = torch.sigmoid(self.video_weight)
            audio_features = audio_weight * audio_features
            video_features = video_weight * video_features
            fused = torch.cat([audio_features, video_features], dim=-1)

        elif self.fusion_type == "attention":
            audio_scores = self.audio_attention(audio_features)
            video_scores = self.video_attention(video_features)
            audio_weights = F.softmax(audio_scores, dim=1)
            video_weights = F.softmax(video_scores, dim=1)
            audio_pooled = (audio_features * audio_weights).sum(dim=1)
            video_pooled = (video_features * video_weights).sum(dim=1)
            fused = audio_pooled + video_pooled
            fused = fused.unsqueeze(1).expand(-1, T, -1)

        elif self.fusion_type == "bilinear":
            fused = self.bilinear(audio_features, video_features)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        fused = self.fusion_layer(fused)

        return fused

    def get_fused_dim(self) -> int:
        """获取融合后的特征维度"""
        return self.fused_dim


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合（带LayerNorm）"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.audio_dim = self.config.AUDIO_ENCODER_FEATURE_DIM
        self.video_dim = self.config.VIDEO_ENCODER_FEATURE_DIM
        self.hidden_dim = self.config.FUSION_DIM

        self.audio_projection = nn.Linear(self.audio_dim, self.hidden_dim)
        self.video_projection = nn.Linear(self.video_dim, self.hidden_dim)

        self.audio_norm = nn.LayerNorm(self.hidden_dim)
        self.video_norm = nn.LayerNorm(self.hidden_dim)

        self.fusion_weights = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 4, 2)
        )

        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, audio_features: torch.Tensor,
               video_features: torch.Tensor) -> torch.Tensor:
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(1)

        B, T_a, D_a = audio_features.shape
        B, T_v, D_v = video_features.shape

        T = min(T_a, T_v)
        audio_features = audio_features[:, :T, :]
        video_features = video_features[:, :T, :]

        audio_proj = self.audio_norm(F.relu(self.audio_projection(audio_features)))
        video_proj = self.video_norm(F.relu(self.video_projection(video_features)))

        concat_features = torch.cat([audio_proj, video_proj], dim=-1)
        weights = F.softmax(self.fusion_weights(concat_features), dim=-1)

        audio_weight = weights[:, :, 0].unsqueeze(-1)
        video_weight = weights[:, :, 1].unsqueeze(-1)

        fused = audio_weight * audio_proj + video_weight * video_proj
        fused = self.dropout(fused)
        fused = self.output_norm(fused)

        return fused


def create_fusion_module(config, fusion_type: str = "simple") -> nn.Module:
    """工厂函数：创建融合模块"""
    if fusion_type == "adaptive":
        return AdaptiveFeatureFusion(config)
    else:
        return FeatureFusion(config, fusion_type)
