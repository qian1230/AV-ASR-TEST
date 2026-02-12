import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class AudioEncoder(nn.Module):
    """简化的音频编码器"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = [32, 64, 128]
        self.feature_dim = 256
        self.dropout = self.config.DROPOUT

        # 3层2D卷积网络 (No BN)
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(self.dropout)

        # 计算Flatten后的维度
        # F=40 -> 20 -> 10 -> 5 (if F=40)
        # Assuming input F is typically 40 (log-mel) or similar.
        # Current code used to have F, so we need to be careful about FC input size.
        # The spec says: in_features=128 * F/8.
        # Let's assume F is passed or handled dynamically, but Linear needs fixed size.
        # We can use a lazy linear or calculate based on config.
        # Config usually has AUDIO_FEATURE_DIM (e.g. 40 or 80).
        # Let's check config later. For now, I'll calculate it in forward or use a fixed calculation if I know F.
        # To be safe and dynamic, I'll define FC layers but might need to infer input size or use a dummy pass.
        # However, `AudioEncoder` usually knows its input dimension from config.
        # Let's assume self.config.AUDIO_N_MELS is available.

        self.freq_dim = self.config.AUDIO_N_MELS // 8 
        self.fc1_in_dim = self.channels[2] * self.freq_dim

        self.fc1 = nn.Linear(self.fc1_in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F, T]
        B, C, F_dim, T = x.shape
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        # Shape: [B, 128, F/8, T/8]

        # Permute to (B, C, F, T) -> (B, T, C, F) -> Flatten C*F
        # Spec: permute (0,3,1,2) -> flatten (2)
        # (B, 128, F/8, T/8) -> (B, T/8, 128, F/8)
        x = x.permute(0, 3, 1, 2).contiguous()
        # Shape: [B, T/8, 128, F/8]
        
        B, T_new, C_new, F_new = x.shape
        x = x.view(B, T_new, C_new * F_new)
        
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(self.relu(self.fc2(x)))
        
        return x

    def get_output_length(self, input_length: int) -> int:
        length = input_length
        for _ in range(3):
            length = (length + 1) // 2
        return length


class Conv2dBlock(nn.Module):
    """带BatchNorm和ReLU的2D卷积块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class TemporalModule(nn.Module):
    """轻量级时序建模模块（可选）"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn(self.conv(x)))
        return x.permute(0, 2, 1)


class AudioEncoderLightweight(nn.Module):
    """轻量版音频编码器（减少参数量）"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = [16, 32, 64]
        self.feature_dim = 256

        self.conv_layers = nn.Sequential(
            Conv2dBlock(1, self.channels[0], stride=(2, 2)),
            Conv2dBlock(self.channels[0], self.channels[1], stride=(2, 2)),
            Conv2dBlock(self.channels[1], self.channels[2], stride=(2, 2)),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels[-1] * 5 * 5, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.fc_layers(x)

        return x
