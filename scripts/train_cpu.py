"""
CPU训练脚本 - 3 epochs
"""
import sys
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

import logging
import time
from configs.cpu_config import load_config
from data.dataset import AVASRDataset
from models.av_asr_model import AVASRModel
from training.trainer import Trainer
def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    """主函数"""
    print("=" * 70)
    print("🚀 AV-ASR 模型训练 (CPU版本)")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. 加载配置
    print("\n📋 步骤1: 加载配置...")
    config = load_config()
    print(f"   设备: {config.DEVICE}")
    print(f"   训练轮数: {config.MAX_EPOCHS}")
    print(f"   批次大小: {config.BATCH_SIZE}")
    print(f"   学习率: {config.LEARNING_RATE}")
    
    # 2. 加载词汇表
    print("\n📖 步骤2: 加载词汇表...")
    vocab = config.get_vocab()
    vocab_size = len(vocab)
    print(f"   词汇表大小: {vocab_size}")
    
    # 3. 创建数据集
    print("\n📊 步骤3: 创建数据集...")
    train_manifest = config.get_manifest_path('train')
    val_manifest = config.get_manifest_path('val')
    test_manifest = config.get_manifest_path('test')
    
    audio_dir = str(config.DATA_DIR / "raw")
    video_dir = str(config.DATA_DIR / "raw")
    
    train_dataset = AVASRDataset(
        manifest_path=train_manifest,
        audio_dir=audio_dir,
        video_dir=video_dir,
        config=config,
        is_training=True
    )
    
    val_dataset = AVASRDataset(
        manifest_path=val_manifest,
        audio_dir=audio_dir,
        video_dir=video_dir,
        config=config,
        is_training=False
    )
    
    test_dataset = AVASRDataset(
        manifest_path=test_manifest,
        audio_dir=audio_dir,
        video_dir=video_dir,
        config=config,
        is_training=False
    )
    
    print(f"   训练集样本数: {len(train_dataset)}")
    print(f"   验证集样本数: {len(val_dataset)}")
    print(f"   测试集样本数: {len(test_dataset)}")
    
    # 4. 创建数据加载器
    print("\n🔄 步骤4: 创建数据加载器...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        collate_fn=val_dataset.collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        collate_fn=test_dataset.collate_fn
    )
    
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader)}")
    print(f"   测试批次数: {len(test_loader)}")
    
    # 5. 创建模型
    print("\n🧠 步骤5: 创建模型...")
    model = AVASRModel(config=config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 6. 创建训练器
    print("\n⚙️  步骤6: 创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )
    
    # 7. 开始训练
    print("\n🏃 步骤7: 开始训练...")
    print(f"   设备: {config.DEVICE}")
    print(f"   最大轮数: {config.MAX_EPOCHS}")
    print("-" * 70)
    
    trainer.train()
    
    # 8. 训练完成
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    print(f"总用时: {elapsed_time/60:.1f} 分钟")
    print(f"检查点保存位置: {config.CHECKPOINT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()
