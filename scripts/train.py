"""
AV-ASR 训练脚本

使用方法:
    python scripts/train.py --config configs/config.py --resume checkpoints/best_model.pth

参数:
    --config: 配置文件路径
    --resume: 从checkpoint恢复训练
    --seed: 随机种子
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn

from configs import config
from models import AVASRModel
from data.dataset import create_dataloaders
from training import Trainer, CTCLoss
from utils.common import set_seed, print_model_summary, get_gpu_memory_usage


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AV-ASR Training")

    parser.add_argument('--config', type=str, default='configs/config.py',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint恢复训练')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批量大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    parser.add_argument('--fusion-type', type=str, default='adaptive',
                       help='融合类型 (simple/weighted/attention/adaptive)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--test-only', action='store_true',
                       help='仅运行测试')

    args = parser.parse_args()

    return args


def setup_experiment(args):
    """设置实验环境"""
    set_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def create_model(args, fusion_type: str = "adaptive"):
    """创建模型"""
    model = AVASRModel(config, fusion_type=fusion_type)

    param_count = model.get_param_count()
    print(f"Model created:")
    print(f"  Total parameters: {param_count['total_params']:,}")
    print(f"  Trainable parameters: {param_count['trainable_params']:,}")

    return model


def train(args):
    """主训练函数"""
    print("=" * 60)
    print("AV-ASR Training")
    print("=" * 60)

    output_dir = setup_experiment(args)

    model = create_model(args, args.fusion_type)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    print(f"\nData loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config
    )

    model_info = trainer.get_model_info()
    print(f"\nTraining configuration:")
    print(f"  Device: {model_info['device']}")
    print(f"  Vocabulary size: {model_info['vocab_size']}")

    gpu_info = get_gpu_memory_usage()
    if gpu_info:
        print(f"  GPU Memory: {gpu_info['allocated_mb']:.1f} MB allocated")

    if args.test_only:
        print("\nRunning test evaluation...")
        test_results = trainer.test()
        return test_results

    print("\nStarting training...")
    start_time = time.time()

    history = trainer.train()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")

    print("\nFinal evaluation on test set...")
    test_results = trainer.test()

    print("\nTraining Summary:")
    print(f"  Best WER: {trainer.best_wer*100:.2f}%")
    print(f"  Test WER: {test_results['wer']*100:.2f}%")
    print(f"  Test Loss: {test_results['loss']:.4f}")

    return history


def main():
    """主函数"""
    args = parse_args()

    if args.epochs:
        config.MAX_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr

    try:
        history = train(args)
        print("\nTraining script completed successfully.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
