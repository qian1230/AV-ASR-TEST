import torch
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

class Config:
    # 项目基础配置
    PROJECT_NAME = "AV-ASR"
    EXPERIMENT_NAME = "av_asr_baseline"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # 音频配置
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MELS = 40
    AUDIO_WINDOW_SIZE = 0.025  # 25ms
    AUDIO_HOP_LENGTH = 0.010   # 10ms
    AUDIO_DURATION_MIN = 1.0
    AUDIO_DURATION_MAX = 5.0

    # 视频配置
    VIDEO_FPS = 30
    VIDEO_SIZE = 128
    VIDEO_GRAYSCALE = True

    # 模型配置
    AUDIO_ENCODER_CHANNELS = [32, 64, 128]
    AUDIO_ENCODER_FEATURE_DIM = 512  # 从256提升到512
    VIDEO_ENCODER_CHANNELS = [32, 64, 128]
    VIDEO_ENCODER_FEATURE_DIM = 512  # 从256提升到512
    FUSION_DIM = 1024  # 512 + 512

    # 词汇表配置（字符级）
    VOCAB_TYPE = "character"
    LOWERCASE = True
    KEEP_SPACES = True
    ADD_BLANK = True
    BLANK_ID = 0

    # 训练配置
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    MAX_EPOCHS = 50
    LEARNING_RATE = 5e-4  # 从1e-4调整
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2  # 从0.1提升
    LABEL_SMOOTHING = 0.1  # 新增

    # 学习率调度
    LR_WARMUP_STEPS = 1000
    LR_SCHEDULER = "cosine"
    LR_MIN = 1e-5
    PATIENCE = 5  # 早停耐心值
    FACTOR = 0.5  # 学习率衰减因子

    # 解码配置
    DECODER_TYPE = "greedy"
    BEAM_SIZE = 5

    # 路径配置
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

    def __init__(self):
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    @classmethod
    def get_vocab(cls):
        """生成字符级词汇表"""
        vocab = ['<blank>', ' ', '!', ',', '.', '?', 'a', 'b', 'c', 'd', 'e',
                 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        return vocab

    @classmethod
    def get_vocab_size(cls):
        """词汇表大小"""
        return len(cls.get_vocab())

config = Config()
