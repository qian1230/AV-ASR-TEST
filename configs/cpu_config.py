"""
AV-ASR 训练配置 - CPU版本（3 epochs）
"""
import torch
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
BASE_DIR = PROJECT_ROOT

class CPUConfig:
    """针对CPU训练的优化配置"""

    # 项目基础配置
    PROJECT_NAME = "AV-ASR"
    EXPERIMENT_NAME = "av_asr_cpu_3epochs"
    DEVICE = "cpu"  # 强制使用CPU
    SEED = 42

    # 音频配置（保持不变）
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MELS = 40
    AUDIO_WINDOW_SIZE = 0.025  # 25ms
    AUDIO_HOP_LENGTH = 0.010   # 10ms
    AUDIO_DURATION_MIN = 1.0
    AUDIO_DURATION_MAX = 5.0

    # 视频配置
    VIDEO_FPS = 30
    VIDEO_SIZE = 64  # 降低分辨率以加快CPU处理
    VIDEO_GRAYSCALE = True

    # 模型配置（轻量级 - MLCA）
    AUDIO_ENCODER_CHANNELS = [32, 64, 128]
    AUDIO_ENCODER_FEATURE_DIM = 256
    VIDEO_ENCODER_CHANNELS = [32, 64, 128]
    VIDEO_ENCODER_FEATURE_DIM = 256
    FUSION_DIM = 512

    # 词汇表配置（从预处理结果读取）
    VOCAB_TYPE = "character"
    LOWERCASE = True
    KEEP_SPACES = True
    ADD_BLANK = True
    BLANK_ID = 0

    # 训练配置（CPU优化）
    BATCH_SIZE = 2  # 小batch size适应CPU
    NUM_WORKERS = 0  # 避免多进程问题
    MAX_EPOCHS = 2   # 用户要求训练3个epoch
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.1

    # 学习率调度
    LR_WARMUP_STEPS = 100
    LR_SCHEDULER = "cosine"
    LR_MIN = 1e-5
    PATIENCE = 10  # 早停耐心值
    FACTOR = 0.5

    # 解码配置
    DECODER_TYPE = "greedy"
    BEAM_SIZE = 5

    # 路径配置
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    @classmethod
    def get_vocab(cls):
        """从预处理生成的词汇表文件加载"""
        vocab_path = cls.DATA_DIR / "vocabulary.json"
        if vocab_path.exists():
            import json
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            return vocab
        else:
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")

    @classmethod
    def get_vocab_size(cls):
        """词汇表大小"""
        return len(cls.get_vocab())

    @classmethod
    def get_manifest_path(cls, split: str) -> str:
        """获取清单文件路径"""
        manifest_file = cls.DATA_DIR / "manifests" / f"{split}_manifest.txt"
        return str(manifest_file)


def load_config() -> CPUConfig:
    """加载配置"""
    return CPUConfig()


# 便捷函数：快速训练配置
def get_quick_config():
    """获取快速训练配置"""
    config = CPUConfig()
    config.MAX_EPOCHS = 3
    config.BATCH_SIZE = 2
    config.NUM_WORKERS = 0
    config.LEARNING_RATE = 1e-3  # 适当提高学习率
    return config


if __name__ == "__main__":
    config = load_config()
    print(f"项目名称: {config.PROJECT_NAME}")
    print(f"实验名称: {config.EXPERIMENT_NAME}")
    print(f"设备: {config.DEVICE}")
    print(f"训练轮数: {config.MAX_EPOCHS}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"词汇表: {config.get_vocab()}")
