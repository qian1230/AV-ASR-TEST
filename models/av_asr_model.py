import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config
from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder, SimpleVideoEncoder
from .fusion import FeatureFusion, AdaptiveFeatureFusion, create_fusion_module


class AVASRModel(nn.Module):
    """AV-ASR 完整模型
    
    采用双模态独立编码 + MLCA融合 + CTC解码
    """

    def __init__(self, config=None, fusion_type: str = "mlca"):
        super().__init__()
        self.config = config if config is not None else globals()['config']
        self.vocab_size = self.config.get_vocab_size()
        self.fusion_type = fusion_type

        self.audio_encoder = AudioEncoder(self.config)

        self.video_encoder = VideoEncoder(self.config)

        # MLCA Fusion
        self.fusion_module = create_fusion_module(self.config, fusion_type="mlca")

        # CTC Head
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化所有模块权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, audio_features: torch.Tensor,
               video_features: torch.Tensor,
               audio_lengths: torch.Tensor = None,
               video_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            audio_features: 音频特征 [B, 1, F, T]
            video_features: 视频特征 [B, T_v, C, H, W]
            audio_lengths: 音频长度 [B]（可选）
            video_lengths: 视频长度 [B]（可选）

        Returns:
            logits: CTC logits [B, T, V]
        """
        # Ensure dimensions
        if audio_features.dim() == 3:
            audio_features = audio_features.unsqueeze(1)
            
        # Video encoder expects [B, T, C, H, W] or [B, C, T, H, W] (handled inside)
        
        audio_encoded = self.audio_encoder(audio_features)
        
        video_encoded = self.video_encoder(video_features)
        
        fused = self.fusion_module(audio_encoded, video_encoded)
        
        logits = self.ctc_head(fused)
        
        # LogSoftmax is applied here or in Loss?
        # Trainer uses nn.CTCLoss which expects LogSoftmax inputs (log_probs).
        # Trainer code: logits = F.log_softmax(logits, dim=-1) inside Loss or explicitly?
        # Let's check Trainer/Loss. 
        # Loss.py: logits = F.log_softmax(logits, dim=-1) is called inside CTCLoss.forward wrapper.
        # But previous model had explicit log_softmax.
        # Spec says: "Output layer ... LogSoftmax".
        # If I add it here, loss.py might do it again.
        # Loss.py: "logits = F.log_softmax(logits, dim=-1)"
        # So I should probably NOT do it here if Loss does it, OR I remove it from Loss.
        # But wait, previous model had:
        # logits = self.ctc_head(fused)
        # logits = torch.nn.functional.log_softmax(logits, dim=-1)
        # And Loss.py also has it. Double log_softmax is bad?
        # Let's check Loss.py again.
        # "logits = F.log_softmax(logits, dim=-1)" in Loss.py.
        # If model outputs log_probs, then log_softmax(log_probs) is not right.
        # I should probably remove it from Model or Loss.
        # Usually Model returns logits (unnormalized) and Loss does log_softmax + ctc_loss.
        # But the previous code had it in Model.
        # Let's stick to returning LOGITS (unnormalized) here, and let Loss handle it.
        # BUT, the Spec says "Output: LogSoftmax".
        # If I strictly follow spec, I should add it.
        # If I add it, I must ensure Loss doesn't double it.
        # Let's return LOGITS for safety and let the Loss handle normalization as it seems robust.
        # However, previous code:
        # logits = torch.nn.functional.log_softmax(logits, dim=-1)
        # return logits
        # Loss.py:
        # logits = F.log_softmax(logits, dim=-1)
        # So it was double log_softmax? log(softmax(log(softmax(x))))?
        # No, log_softmax is stable.
        # Let's check if Loss.py checks for log_softmax.
        # Actually, `nn.CTCLoss` expects Log Probs.
        # If I return logits, Loss.py does log_softmax -> Correct.
        # If I return log_probs, Loss.py does log_softmax(log_probs) -> Incorrect (it would be log_softmax again).
        # So I will return LOGITS here.
        
        return logits

    def forward_with_lengths(self, audio_features: torch.Tensor,
                           video_features: torch.Tensor,
                           audio_lengths: torch.Tensor,
                           video_lengths: torch.Tensor) -> dict:
        """
        带长度信息的前向传播（用于CTCLoss计算）

        Args:
            audio_features: 音频特征 [B, 1, F, T]
            video_features: 视频特征 [B, T_v, C, H, W]
            audio_lengths: 音频长度 [B]
            video_lengths: 视频长度 [B]

        Returns:
            dict: 包含logits和有效长度的字典
        """
        audio_features = audio_features.unsqueeze(1) if audio_features.dim() == 3 else audio_features

        if video_features.dim() == 4:
            video_features = video_features.unsqueeze(1)

        audio_encoded = self.audio_encoder(audio_features)
        B_a, T_a, D_a = audio_encoded.shape

        video_encoded = self.video_encoder(video_features)
        B_v, T_v, D_v = video_encoded.shape

        T = min(T_a, T_v)

        audio_encoded = audio_encoded[:, :T, :]
        video_encoded = video_encoded[:, :T, :]

        fused = self.fusion_module(audio_encoded, video_encoded)

        logits = self.ctc_head(fused)

        output_lengths = torch.full((B_a,), T, dtype=torch.long, device=logits.device)

        return {
            'logits': logits,
            'output_lengths': output_lengths
        }

    def get_param_count(self) -> dict:
        """获取参数量统计"""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """静态方法：计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AVASRModelLightweight(nn.Module):
    """轻量版AV-ASR模型（参数量更少）"""

    def __init__(self, config=None, fusion_type: str = "simple"):
        super().__init__()
        self.config = config or config
        self.vocab_size = self.config.get_vocab_size()

        from .audio_encoder import AudioEncoderLightweight
        from .video_encoder import SimpleVideoEncoder

        self.audio_encoder = AudioEncoderLightweight(self.config)
        self.video_encoder = SimpleVideoEncoder(self.config)

        if fusion_type == "simple":
            fused_dim = self.config.AUDIO_ENCODER_FEATURE_DIM + self.config.VIDEO_ENCODER_FEATURE_DIM
        else:
            fused_dim = self.config.AUDIO_ENCODER_FEATURE_DIM

        self.fusion = create_fusion_module(self.config, fusion_type)

        self.ctc_head = nn.Sequential(
            nn.Linear(fused_dim, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, audio_features: torch.Tensor,
               video_features: torch.Tensor) -> torch.Tensor:
        audio_features = audio_features.unsqueeze(1) if audio_features.dim() == 3 else audio_features

        if video_features.dim() == 4:
            video_features = video_features.unsqueeze(1)

        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)

        fused = self.fusion(audio_encoded, video_encoded)
        logits = self.ctc_head(fused)

        return logits


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块（实验性）"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config

        self.scales = [1, 2, 4]

        self.audio_projectors = nn.ModuleList([
            nn.Linear(config.AUDIO_ENCODER_FEATURE_DIM, 256) for _ in self.scales
        ])

        self.video_projectors = nn.ModuleList([
            nn.Linear(config.VIDEO_ENCODER_FEATURE_DIM, 256) for _ in self.scales
        ])

        self.fusion_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

        self.output_projector = nn.Linear(512, 512)

    def forward(self, audio_features: torch.Tensor,
               video_features: torch.Tensor) -> torch.Tensor:
        B, T, _ = audio_features.shape

        multi_scale_audio = []
        multi_scale_video = []

        for scale, audio_proj, video_proj in zip(
            self.scales, self.audio_projectors, self.video_projectors
        ):
            if scale == 1:
                audio_scale = audio_proj(audio_features)
                video_scale = video_proj(video_features)
            else:
                audio_scale = F.interpolate(
                    audio_features.permute(0, 2, 1).unsqueeze(-1),
                    size=(T // scale, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(-1).permute(0, 2, 1)
                video_scale = F.interpolate(
                    video_features.permute(0, 2, 1).unsqueeze(-1),
                    size=(T // scale, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(-1).permute(0, 2, 1)
                audio_scale = audio_proj(audio_scale)
                video_scale = video_proj(video_scale)

            multi_scale_audio.append(audio_scale)
            multi_scale_video.append(video_scale)

        audio_concat = torch.cat(multi_scale_audio, dim=1)
        video_concat = torch.cat(multi_scale_video, dim=1)

        attention_weights = self.fusion_attention(
            torch.cat([audio_concat, video_concat], dim=-1)
        )
        audio_weight = attention_weights[:, :, 0:1]
        video_weight = attention_weights[:, :, 1:2]

        fused = audio_weight * audio_concat + video_weight * video_concat
        fused = self.output_projector(fused)

        return fused
