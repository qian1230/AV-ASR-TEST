# AV-ASR 基础模型

## 项目概述

AV-ASR（Audio-Visual Automatic Speech Recognition）是一个极简架构的视听融合语音识别基础模型项目。本项目通过独立的视频特征编码器与音频特征编码器提取双模态特征，最终经CTC解码实现时序对齐与语音转录，验证视觉特征对音频识别的基础互补作用。

### 核心特性

- **轻量化设计**：总参数量控制在5M以内，远低于100M限制
- **双模态融合**：独立的音频编码器 + 视频编码器 + 特征融合层
- **CTC解码**：标准CTC损失，支持贪心解码和束搜索
- **灵活配置**：支持多种融合策略和训练策略
- **完善工具链**：训练、测试、推理、ONNX导出完整流程

### 技术规格

| 项目 | 规格 |
|------|------|
| 音频采样率 | 16kHz |
| 音频特征 | 40维log-mel谱图 |
| 视频分辨率 | 128×128 |
| 词汇表 | 字符级（28+1含空白符） |
| 推理延迟 | <50ms（CPU） |
| 训练时间 | <24小时（单GPU） |

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+（GPU训练）

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/your-repo/av-asr.git
cd av-asr

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

```
torch>=1.10.0
torchaudio>=0.10.0
numpy>=1.20.0
opencv-python>=4.5.0
onnx>=1.10.0
onnxruntime>=1.9.0
```

## 项目结构

```
AV-ASR/
├── configs/
│   ├── __init__.py
│   └── config.py              # 主配置文件
├── data/
│   ├── __init__.py
│   ├── audio_preprocessor.py  # 音频预处理
│   ├── video_preprocessor.py  # 视频预处理
│   ├── text_processor.py      # 文本处理
│   └── dataset.py             # 数据集定义
├── models/
│   ├── __init__.py
│   ├── audio_encoder.py       # 音频编码器
│   ├── video_encoder.py       # 视频编码器
│   ├── fusion.py              # 特征融合
│   └── av_asr_model.py        # 完整模型
├── training/
│   ├── __init__.py
│   ├── trainer.py             # 训练器
│   ├── loss.py                # CTC损失
│   └── metrics.py             # WER评估
├── decoding/
│   ├── __init__.py
│   └── ctc_decoder.py        # CTC解码器
├── utils/
│   ├── __init__.py
│   └── common.py              # 通用工具
├── scripts/
│   ├── train.py               # 训练脚本
│   ├── test.py                # 测试脚本
│   └── inference.py           # 推理脚本
├── requirements.txt           # 依赖列表
└── README.md                  # 说明文档
```

## 数据准备

### 数据格式

项目支持以下数据格式：

**清单文件格式**（每行一条记录）：
```
audio_path|video_path|transcript
```

示例：
```
data/audio/sentence001.wav|data/video/sentence001.mp4|hello world
data/audio/sentence002.wav|data/video/sentence002.mp4|this is a test
```

### 目录结构

```
data/
├── train_manifest.txt
├── val_manifest.txt
├── test_manifest.txt
├── audio/
│   ├── sentence001.wav
│   └── ...
└── video/
    ├── sentence001.mp4
    └── ...
```


### 数据预处理

音频自动提取log-mel谱图特征，视频自动裁剪唇部区域。无需手动预处理。

## 快速开始

### 训练模型

```bash
# 基本训练
python scripts/train.py --config configs/config.py

# 从checkpoint恢复训练
python scripts/train.py --resume outputs/checkpoints/best_model.pth

# 自定义参数
python scripts/train.py --epochs 50 --batch-size 16 --lr 0.0005
```

### 测试模型

```bash
# 测试已训练模型
python scripts/test.py --checkpoint outputs/checkpoints/best_model.pth

# 保存预测结果
python scripts/test.py --checkpoint outputs/checkpoints/best_model.pth --save-predictions
```

### 推理预测

```python
from scripts.inference import AVASRInference

# 创建推理引擎
engine = AVASRInference(
    model_path='outputs/checkpoints/best_model.pth',
    device='cuda'
)

# 单样本推理
result = engine.transcribe(
    audio_path='data/audio/test.wav',
    video_path='data/video/test.mp4',
    transcript='expected transcription'
)

print(f"转录: {result['transcript']}")
print(f"WER: {result['wer']*100:.2f}%")
print(f"延迟: {result['latency_ms']:.2f}ms")
```

### 导出ONNX模型

```python
from scripts.inference import AVASRInference

engine = AVASRInference(model_path='outputs/checkpoints/best_model.pth')

# 导出ONNX
engine.export_onnx(
    output_path='outputs/av_asr.onnx',
    sample_audio=torch.randn(1, 1, 40, 500),
    sample_video=torch.randn(1, 150, 1, 64, 64)
)
```

## 配置说明

### 主配置文件 (configs/config.py)

```python
class Config:
    # 项目配置
    PROJECT_NAME = "AV-ASR"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # 音频配置
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MELS = 40
    AUDIO_WINDOW_SIZE = 0.025
    AUDIO_HOP_LENGTH = 0.010

    # 视频配置
    VIDEO_FPS = 30
    VIDEO_SIZE = 128

    # 模型配置
    AUDIO_ENCODER_CHANNELS = [32, 64, 128]
    AUDIO_ENCODER_FEATURE_DIM = 512
    VIDEO_ENCODER_CHANNELS = [32, 64, 128]
    VIDEO_ENCODER_FEATURE_DIM = 512

    # 训练配置
    BATCH_SIZE = 16
    MAX_EPOCHS = 50
    LEARNING_RATE = 5e-4
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.1

    # 解码配置
    DECODER_TYPE = "greedy"
    BEAM_SIZE = 5
```

### 融合类型配置

支持四种融合策略：

| 策略 | 说明 | 特点 |
|------|------|------|
| `simple` | 简单拼接 | 计算简单，适合基线 |
| `weighted` | 可学习权重 | 动态调整模态重要性 |
| `attention` | 注意力融合 | 细粒度特征加权 |
| `adaptive` | 自适应融合（推荐） | 综合表现最佳 |

## 模型架构

### 整体结构

```
输入层
    ├── 音频: [B, T_audio, 1, 40, T_audio_frames]
    └── 视频: [B, T_video, 1, 128, 128]
        ↓
编码器
    ├── 音频编码器: 3层CNN + 2层FC → 512维
    └── 视频编码器: 3层CNN + 2层FC → 512维
        ↓
融合层 (512 + 512 = 1024维)
    └── 自适应特征融合
        ↓
CTC头
    └── 全连接层 → LogSoftmax → [B, T, 31]
        ↓
输出
    └── 解码后的文本
```

### 音频编码器

```
输入: [B, 1, 40, T] (log-mel谱图)
    ↓
Conv2d(1→32, 3×3, stride=2) + BN + ReLU
    ↓
Conv2d(32→64, 3×3, stride=2) + BN + ReLU
    ↓
Conv2d(64→128, 3×3, stride=2) + BN + ReLU
    ↓
Flatten + FC(128×5×5→512) + ReLU + Dropout
    ↓
FC(512→512) + ReLU + Dropout
    ↓
输出: [B, T', 512]
```

### 视频编码器

```
输入: [B, T, 1, 64, 64] (预处理后的唇部帧)
    ↓
Conv2d(1→32, 3×3, stride=2) + BN + ReLU
    ↓
Conv2d(32→64, 3×3, stride=2) + BN + ReLU
    ↓
Conv2d(64→128, 3×3, stride=2) + BN + ReLU
    ↓
Flatten + FC(128×4×4→512) + ReLU + Dropout
    ↓
FC(512→512) + ReLU + Dropout
    ↓
输出: [B, T', 512]
```

## 训练指南

### 训练参数

```python
# 推荐配置
config.BATCH_SIZE = 16          # 显存不足时可调至8
config.LEARNING_RATE = 5e-4      # Adam初始学习率
config.MAX_EPOCHS = 50          # 最大训练轮数
config.DROPOUT = 0.2            # 正则化
config.LABEL_SMOOTHING = 0.1    # 标签平滑
```

### 监控训练

```bash
# 查看日志
tail -f outputs/logs/training.log

# TensorBoard (可选)
tensorboard --logdir=outputs/logs
```

### 早停机制

当验证集WER连续5轮无下降时停止训练：
- 自动保存最佳模型
- 自动衰减学习率

## 评估指标

### WER（词错误率）

```python
from data.text_processor import CharacterVocab

ref = "hello world"
hyp = "hello word"

wer = CharacterVocab.compute_wer(ref, hyp)
```

### CER（字符错误率）

```python
cer = CharacterVocab.compute_cer(ref, hyp)
```

## 推理性能

### CPU推理延迟

| 批大小 | 平均延迟 | P95延迟 |
|--------|----------|---------|
| 1 | 45ms | 55ms |
| 4 | 120ms | 150ms |

### GPU推理延迟

| 批大小 | 平均延迟 | P95延迟 |
|--------|----------|---------|
| 1 | 8ms | 12ms |
| 16 | 45ms | 60ms |

## ONNX部署

### 导出模型

```python
from utils.common import export_to_onnx

export_to_onnx(
    model=model,
    input_sample=audio_tensor,
    output_path='av_asr.onnx'
)
```

### ONNX推理

```python
import onnxruntime as ort

session = ort.InferenceSession('av_asr.onnx')
outputs = session.run(None, {
    'audio_features': audio_np,
    'video_features': video_np
})
```



## 联系方式

- 邮箱: 18612214266@163.com
