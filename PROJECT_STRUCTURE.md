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
│   └── av_asr_model.py        # 完整AV-ASR模型
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
├── requirements.txt           # 依赖
└── README.md                  # 说明文档
