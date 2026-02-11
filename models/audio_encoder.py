import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class AudioEncoder(nn.Module):
    """简化的音频编码器"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = self.config.AUDIO_ENCODER_CHANNELS
        self.feature_dim = self.config.AUDIO_ENCODER_FEATURE_DIM
        self.dropout = self.config.DROPOUT

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, freq_dim, T = x.shape

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        B, C, freq_dim, T = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * freq_dim)

        return x

    def get_output_length(self, input_length: int) -> int:
        return input_length


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
