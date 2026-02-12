import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class VideoEncoder(nn.Module):
    """视频编码器：3层2D卷积 + 2层全连接层

    设计决策：
    - 与音频编码器对称设计
    - 使用BatchNorm提升训练稳定性
    - 特征维度从256提升到512
    - 输入为预处理后的唇部区域序列
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = [32, 64, 128]
        self.feature_dim = 256
        self.dropout = self.config.DROPOUT

        # 3层3D卷积网络 (No BN)
        self.conv1 = nn.Conv3d(1, self.channels[0], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Global Average Pooling on Spatial Dimensions (H, W)
        # Output will be (B, 128, T, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # FC Layers
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> need [B, C, T, H, W] for Conv3d
        # If input is [B, T, C, H, W], permute to [B, C, T, H, W]
        if x.dim() == 5:
            x = x.permute(0, 2, 1, 3, 4)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Pooling spatial dimensions
        x = self.pool(x)
        # Shape: [B, 128, T, 1, 1]
        
        # Flatten
        # Permute to [B, T, 128]
        x = x.flatten(3) # [B, 128, T, 1]
        x = x.squeeze(-1) # [B, 128, T]
        x = x.permute(0, 2, 1) # [B, T, 128]
        
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(self.relu(self.fc2(x)))
        
        return x

    def get_output_length(self, input_length: int) -> int:
        """计算输出序列长度"""
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


class VideoEncoderWithTemporal(nn.Module):
    """带时序建模的视频编码器"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = [32, 64, 128]
        self.feature_dim = 512

        self.spatial_encoder = nn.Sequential(
            Conv2dBlock(1, self.channels[0], stride=(2, 2)),
            Conv2dBlock(self.channels[0], self.channels[1], stride=(2, 2)),
            Conv2dBlock(self.channels[1], self.channels[2], stride=(2, 2)),
            Conv2dBlock(self.channels[2], self.channels[2], stride=(1, 1), padding=1),
        )

        self.temporal_encoder = nn.ModuleList([
            nn.LSTM(self.channels[-1], self.channels[-1] // 2, batch_first=True, bidirectional=True)
            for _ in range(1)
        ])

        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels[-1], self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.spatial_encoder(x)
        x = x.mean(dim=[2, 3])
        x = x.view(B, T, -1)

        for lstm in self.temporal_encoder:
            x, _ = lstm(x)

        x = self.fc_layers(x)

        return x


class SimpleVideoEncoder(nn.Module):
    """简单视频编码器（无时序建模）"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config
        self.channels = [16, 32, 64]
        self.feature_dim = 256

        self.encoder = nn.Sequential(
            Conv2dBlock(1, self.channels[0], stride=(2, 2)),
            Conv2dBlock(self.channels[0], self.channels[1], stride=(2, 2)),
            Conv2dBlock(self.channels[1], self.channels[2], stride=(2, 2)),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(self.channels[-1], self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return x.view(B, T, -1)
