import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class MLCAFusion(nn.Module):
    """Multi-Scale Cross-Modal Attention (MLCA) Fusion Module"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.audio_dim = 256  # As per spec
        self.video_dim = 256  # As per spec
        self.fused_dim = 512

        # Multi-scale feature mapping (Audio)
        self.audio_conv3 = nn.Conv1d(self.audio_dim, 128, kernel_size=3, padding=1)
        self.audio_conv5 = nn.Conv1d(self.audio_dim, 128, kernel_size=5, padding=2)

        # Multi-scale feature mapping (Video)
        self.video_conv3 = nn.Conv1d(self.video_dim, 128, kernel_size=3, padding=1)
        self.video_conv5 = nn.Conv1d(self.video_dim, 128, kernel_size=5, padding=2)

        # Output projection
        self.output_proj = nn.Linear(512, 512)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _temporal_alignment(self, audio_features, video_features):
        """
        Align temporal dimensions using linear interpolation/downsampling.
        audio: [B, T_a, D]
        video: [B, T_v, D]
        """
        B, T_a, D_a = audio_features.shape
        _, T_v, D_v = video_features.shape

        # Target length T = max(T_a, T_v) -> As per spec: "取 T_a、T_v 均值" 
        # Wait, spec says: "取 T_a、T_v 均值" (mean of Ta and Tv)? 
        # Let's re-read: "取 T_a、T_v 均值" (take mean of Ta and Tv).
        # T = (T_a + T_v) // 2
        
        target_T = (T_a + T_v) // 2
        if target_T == 0: target_T = 1

        # Permute for interpolate: [B, D, T]
        audio_in = audio_features.permute(0, 2, 1)
        video_in = video_features.permute(0, 2, 1)

        audio_aligned = F.interpolate(audio_in, size=target_T, mode='linear', align_corners=False)
        video_aligned = F.interpolate(video_in, size=target_T, mode='linear', align_corners=False)

        # Permute back: [B, T, D]
        return audio_aligned.permute(0, 2, 1), video_aligned.permute(0, 2, 1)

    def forward(self, audio_features, video_features):
        # 1. Temporal Alignment
        audio_features, video_features = self._temporal_alignment(audio_features, video_features)
        
        # Shape: [B, T, 256]
        B, T, _ = audio_features.shape

        # 2. Multi-scale feature mapping
        # Input to Conv1d needs [B, C, T]
        a_in = audio_features.permute(0, 2, 1) # [B, 256, T]
        v_in = video_features.permute(0, 2, 1) # [B, 256, T]

        a_conv3 = F.gelu(self.audio_conv3(a_in)) # [B, 128, T]
        a_conv5 = F.gelu(self.audio_conv5(a_in)) # [B, 128, T]
        
        v_conv3 = F.gelu(self.video_conv3(v_in)) # [B, 128, T]
        v_conv5 = F.gelu(self.video_conv5(v_in)) # [B, 128, T]

        # Concatenate multi-scale features
        a_multi = torch.cat([a_conv3, a_conv5], dim=1) # [B, 256, T]
        v_multi = torch.cat([v_conv3, v_conv5], dim=1) # [B, 256, T]

        # Back to [B, T, 256] for attention
        a_multi = a_multi.permute(0, 2, 1)
        v_multi = v_multi.permute(0, 2, 1)

        # 3. Cross-modal Attention
        # Cosine Similarity: [B, T, T]
        # Normalize vectors for cosine similarity
        a_norm = F.normalize(a_multi, p=2, dim=-1)
        v_norm = F.normalize(v_multi, p=2, dim=-1)
        
        similarity = torch.bmm(a_norm, v_norm.transpose(1, 2)) # [B, T, T]

        # Attention weights
        attn_a2v = F.softmax(similarity, dim=-1) # Audio attends to Video
        attn_v2a = F.softmax(similarity, dim=-2) # Video attends to Audio (transpose logic? usually dim=-1 of transpose, effectively dim=-2)
        # Wait, if we want V features for A, we do A x V_T.
        # A_enhanced = Softmax(A x V_T) * V
        # V_enhanced = Softmax(V x A_T) * A = Softmax((A x V_T)^T) * A
        
        # Audio enhanced by Video
        a_enhanced = torch.bmm(attn_a2v, v_multi) # [B, T, 256]
        
        # Video enhanced by Audio
        # We need attention map where rows are video frames and cols are audio frames.
        # similarity is [B, T_a, T_v] (here T_a=T_v=T). row i is audio frame i.
        # We want for video frame j, weights over audio frames.
        # That corresponds to columns of similarity matrix.
        # So we softmax over dim 1 (audio frames).
        attn_v2a = F.softmax(similarity, dim=1).transpose(1, 2) # [B, T, T] -> transpose to match [B, T_v, T_a]
        
        v_enhanced = torch.bmm(attn_v2a, a_multi) # [B, T, 256]

        # 4. Residual Connection & Fusion
        a_out = audio_features + a_enhanced
        v_out = video_features + v_enhanced

        fused = torch.cat([a_out, v_out], dim=-1) # [B, T, 512]
        
        fused = F.gelu(self.output_proj(fused))
        fused = self.dropout(fused)

        return fused


class FeatureFusion(nn.Module):
    """Wrapper to maintain compatibility or dispatch"""
    def __init__(self, config=None, fusion_type="mlca"):
        super().__init__()
        self.config = config or config
        if fusion_type == "mlca":
            self.impl = MLCAFusion(config)
            self.fused_dim = 512
        else:
            # Fallback or other types if needed, but for now we focus on MLCA
            self.impl = MLCAFusion(config)
            self.fused_dim = 512

    def forward(self, audio, video):
        return self.impl(audio, video)


class AdaptiveFeatureFusion(FeatureFusion):
    def __init__(self, config=None):
        super().__init__(config, fusion_type="mlca")


def create_fusion_module(config, fusion_type="mlca"):
    return FeatureFusion(config, fusion_type)
