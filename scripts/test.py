"""
AV-ASR 测试脚本

使用方法:
    python scripts/test.py --checkpoint checkpoints/best_model.pth --data-dir data

参数:
    --checkpoint: 模型checkpoint路径
    --data-dir: 数据目录
    --batch-size: 批量大小
    --output: 输出结果文件
"""
import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from configs import config
from models import AVASRModel
from data.dataset import create_dataloaders, AVASRDataset
from data.text_processor import TextProcessor, CharacterVocab
from training.metrics import WERCalculator, CERCalculator, RunningMetrics
from decoding.ctc_decoder import CTCDecoder
from utils.common import set_seed, get_model_size, get_gpu_memory_usage


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AV-ASR Testing")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='输出结果文件')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--fusion-type', type=str, default='adaptive',
                       help='融合类型')
    parser.add_argument('--save-predictions', action='store_true',
                       help='保存预测结果')

    args = parser.parse_args()
    return args


def load_model(checkpoint_path: str, fusion_type: str = "adaptive") -> AVASRModel:
    """加载模型"""
    model = AVASRModel(config, fusion_type=fusion_type)
    model = model.to(config.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Model size: {get_model_size(model):.2f} MB")

    model.eval()
    return model


def evaluate_test_set(model: nn.Module,
                     test_loader,
                     text_processor: TextProcessor) -> Dict:
    """评估测试集"""
    print("=" * 60)
    print("EVALUATING TEST SET")
    print("=" * 60)

    model.eval()
    wer_calculator = WERCalculator(config)
    cer_calculator = CERCalculator(config)
    metrics = RunningMetrics()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            audio_features = batch['audio_features'].to(config.DEVICE)
            video_features = batch['video_features'].to(config.DEVICE)
            targets = batch['targets'].to(config.DEVICE)
            target_lengths = batch['target_lengths'].to(config.DEVICE)

            logits = model(audio_features, video_features)

            decoder = CTCDecoder(config)
            decoder.idx_to_char = text_processor.idx_to_char

            predictions = decoder.greedy_decode(logits)

            for i in range(len(predictions)):
                hyp = text_processor.decode(predictions[i])
                ref = batch['transcript'][i]

                all_predictions.append(hyp)
                all_references.append(ref)

                wer = wer_calculator.compute_wer(ref, hyp)
                cer = cer_calculator.compute_cer(ref, hyp)

                metrics.update(0, wer, cer, 1)

            if (batch_idx + 1) % 10 == 0:
                current_metrics = metrics.compute()
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches | "
                      f"WER: {current_metrics['wer']*100:.2f}%")

    final_metrics = metrics.compute()

    print(f"\nTest Results:")
    print(f"  WER: {final_metrics['wer']*100:.2f}%")
    print(f"  CER: {final_metrics['cer']*100:.2f}%")
    print(f"  Total samples: {len(all_predictions)}")

    detailed_results = {
        'predictions': all_predictions,
        'references': all_references,
        'metrics': {
            'wer': final_metrics['wer'],
            'cer': final_metrics['cer'],
            'num_samples': len(all_predictions)
        }
    }

    return detailed_results


def evaluate_single_sample(model: nn.Module,
                          audio_features: torch.Tensor,
                          video_features: torch.Tensor,
                          reference: str,
                          text_processor: TextProcessor) -> Dict:
    """评估单个样本"""
    model.eval()

    with torch.no_grad():
        logits = model(audio_features, video_features)

    decoder = CTCDecoder(config)
    decoder.idx_to_char = text_processor.idx_to_char

    prediction = decoder.greedy_decode(logits)
    hypothesis = text_processor.decode(prediction)

    wer = CharacterVocab.compute_wer(reference, hypothesis)
    cer = CharacterVocab.compute_cer(reference, hypothesis)

    return {
        'reference': reference,
        'hypothesis': hypothesis,
        'wer': wer,
        'cer': cer
    }


def print_sample_predictions(predictions: List[Dict],
                           num_samples: int = 10):
    """打印样本预测结果"""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    for i, sample in enumerate(predictions[:num_samples]):
        wer_status = "✓" if sample['wer'] < 0.2 else "✗"
        print(f"\nSample {i+1}:")
        print(f"  Reference: {sample['reference']}")
        print(f"  Hypothesis: {sample['hypothesis']}")
        print(f"  WER: {sample['wer']*100:.1f}% {wer_status}")
        print(f"  CER: {sample['cer']*100:.1f}%")


def save_results(results: Dict, output_path: str):
    """保存结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


def measure_inference_latency(model: nn.Module,
                             num_samples: int = 100) -> Dict:
    """测量推理延迟"""
    print("\n" + "=" * 60)
    print("LATENCY MEASUREMENT")
    print("=" * 60)

    model.eval()

    dummy_audio = torch.randn(1, 1, config.AUDIO_N_MELS, 500).to(config.DEVICE)
    dummy_video = torch.randn(1, 150, 1, 64, 64).to(config.DEVICE)

    latencies = []

    with torch.no_grad():
        for i in range(num_samples + 20):
            start = time.time()
            _ = model(dummy_audio, dummy_video)
            latency = time.time() - start

            if i >= 20:
                latencies.append(latency * 1000)

    latencies_ms = latencies
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    std_latency = (sum((x - mean_latency) ** 2 for x in latencies_ms) / len(latencies_ms)) ** 0.5

    print(f"Number of samples: {len(latencies_ms)}")
    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"Std latency: {std_latency:.2f} ms")
    print(f"Min latency: {min(latencies_ms):.2f} ms")
    print(f"Max latency: {max(latencies_ms):.2f} ms")

    return {
        'mean_ms': mean_latency,
        'std_ms': std_latency,
        'min_ms': min(latencies_ms),
        'max_ms': max(latencies_ms)
    }


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("AV-ASR TESTING")
    print("=" * 60)

    set_seed(args.seed)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {config.DEVICE}")

    gpu_info = get_gpu_memory_usage()
    if gpu_info:
        print(f"GPU: {gpu_info['device']}")

    model = load_model(args.checkpoint, args.fusion_type)

    test_loader = create_test_dataloader(args.data_dir, args.batch_size)

    text_processor = TextProcessor(config)

    print("\nRunning evaluation...")
    start_time = time.time()

    results = evaluate_test_set(model, test_loader, text_processor)

    evaluation_time = time.time() - start_time
    print(f"\nEvaluation completed in {evaluation_time:.1f} seconds")

    latency_results = measure_inference_latency(model)

    if args.save_predictions:
        print_sample_predictions(
            [{'reference': r, 'hypothesis': h, 'wer': 0, 'cer': 0}
             for r, h in zip(results['references'], results['predictions'])],
            10
        )

    final_results = {
        'checkpoint': args.checkpoint,
        'metrics': results['metrics'],
        'latency': latency_results,
        'evaluation_time_seconds': evaluation_time
    }

    save_results(final_results, args.output)

    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)

    return final_results


def create_test_dataloader(data_dir: str, batch_size: int):
    """创建测试数据加载器"""
    data_path = Path(data_dir)
    test_manifest = data_path / "test_manifest.txt"

    dataset = AVASRDataset(
        manifest_path=str(test_manifest) if test_manifest.exists() else "",
        audio_dir=str(data_path / "audio"),
        video_dir=str(data_path / "video"),
        config=config,
        is_training=False
    )

    from data.audio_preprocessor import DynamicPaddingCollator
    collator = DynamicPaddingCollator(pad_value=0.0)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True
    )

    return loader


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
