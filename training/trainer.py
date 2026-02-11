import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from configs import config
from models import AVASRModel
from training.loss import CTCLoss, LabelSmoothingCTCLoss
from training.metrics import WERCalculator, CERCalculator, RunningMetrics
from data.text_processor import TextProcessor


class Trainer:
    """AV-ASR模型训练器

    功能：
    - 单GPU/CPU训练
    - 训练/验证/测试循环
    - Checkpoint保存与加载
    - 早停机制
    - 学习率调度
    - 训练日志记录
    """

    def __init__(self,
                 model: AVASRModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config=None,
                 text_processor: TextProcessor = None):
        """
        Args:
            model: AV-ASR模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
            text_processor: 文本处理器
        """
        self.config = config or config
        self.model = model.to(self.config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.text_processor = text_processor or TextProcessor(self.config)

        self.criterion = CTCLoss(self.config)
        self.wer_calculator = WERCalculator(self.config)
        self.cer_calculator = CERCalculator(self.config)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.start_epoch = 0
        self.best_wer = float('inf')
        self.patience_counter = 0

        self.checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_metrics = RunningMetrics()
        self.val_metrics = RunningMetrics()

        self.log_file = self.config.LOG_DIR / "training.log"

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.LR_SCHEDULER == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.FACTOR,
                patience=self.config.PATIENCE // 2,
                min_lr=self.config.LR_MIN,
                verbose=True
            )
        elif self.config.LR_SCHEDULER == "cosine":
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.LR_MIN
            )
        else:
            scheduler = None

        return scheduler

    def train(self) -> Dict[str, Any]:
        """执行完整训练流程"""
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Epochs: {self.config.MAX_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print("-" * 50)

        training_history = {
            'train_loss': [],
            'train_wer': [],
            'val_loss': [],
            'val_wer': [],
            'learning_rates': [],
            'epochs': []
        }

        for epoch in range(self.start_epoch, self.config.MAX_EPOCHS):
            epoch_start = time.time()

            train_loss, train_wer = self._train_epoch(epoch)
            val_loss, val_wer = self._validate_epoch(epoch)

            training_history['train_loss'].append(train_loss)
            training_history['train_wer'].append(train_wer)
            training_history['val_loss'].append(val_loss)
            training_history['val_wer'].append(val_wer)
            training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            training_history['epochs'].append(epoch)

            epoch_time = time.time() - epoch_start

            print(f"Epoch {epoch+1}/{self.config.MAX_EPOCHS} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Train WER: {train_wer*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val WER: {val_wer*100:.2f}%")

            self._log_training(epoch, train_loss, train_wer, val_loss, val_wer)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_wer < self.best_wer:
                self.best_wer = val_wer
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  -> New best WER: {self.best_wer*100:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, is_best=False)

        self._save_final_checkpoint()
        self._save_training_history(training_history)

        return training_history

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练单个epoch"""
        self.model.train()
        self.train_metrics.reset()

        current_lr = self.optimizer.param_groups[0]['lr']

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            audio_features = batch['audio_features'].to(self.config.DEVICE)
            video_features = batch['video_features'].to(self.config.DEVICE)
            targets = batch['targets'].to(self.config.DEVICE)
            target_lengths = batch['target_lengths'].to(self.config.DEVICE)
            audio_lengths = batch['audio_lengths'].to(self.config.DEVICE)
            video_lengths = batch['video_lengths'].to(self.config.DEVICE)

            output_lengths = torch.full(
                (audio_features.size(0),),
                audio_features.size(-1) // 8,
                dtype=torch.long,
                device=self.config.DEVICE
            )

            try:
                logits = self.model(
                    audio_features, video_features,
                    audio_lengths, video_lengths
                )

                if logits.dim() == 3:
                    output_lengths = torch.full(
                        (audio_features.size(0),),
                        logits.size(1),
                        dtype=torch.long,
                        device=self.config.DEVICE
                    )
                    loss = self.criterion(
                        logits,
                        targets,
                        output_lengths,
                        target_lengths
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                with torch.no_grad():
                    hypotheses = self._decode_batch(logits)
                    references = [batch['transcript'][i] for i in range(len(batch['transcript']))]

                    batch_wer = self.wer_calculator.compute_batch_wer(references, hypotheses)
                    batch_cer = self.cer_calculator.compute_batch_cer(references, hypotheses)

                    self.train_metrics.update(
                        loss.item(),
                        batch_wer['wer'],
                        batch_cer['cer'],
                        len(references)
                    )

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Skipping batch {batch_idx} due to OOM")
                    continue
                else:
                    raise e

        train_metrics = self.train_metrics.compute()
        return train_metrics['loss'], train_metrics['wer']

    def _validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """验证单个epoch"""
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                audio_features = batch['audio_features'].to(self.config.DEVICE)
                video_features = batch['video_features'].to(self.config.DEVICE)
                targets = batch['targets'].to(self.config.DEVICE)
                target_lengths = batch['target_lengths'].to(self.config.DEVICE)
                audio_lengths = batch['audio_lengths'].to(self.config.DEVICE)
                video_lengths = batch['video_lengths'].to(self.config.DEVICE)

                try:
                    logits = self.model(
                        audio_features, video_features,
                        audio_lengths, video_lengths
                    )

                    if logits.dim() == 3:
                        output_lengths = torch.full(
                            (audio_features.size(0),),
                            logits.size(1),
                            dtype=torch.long,
                            device=self.config.DEVICE
                        )
                    else:
                        output_lengths = torch.full(
                            (audio_features.size(0),),
                            logits.size(1), # Usually not used for CE loss but for consistency
                            dtype=torch.long,
                            device=self.config.DEVICE
                        )

                    loss = self.criterion(
                        logits,
                        targets,
                        output_lengths,
                        target_lengths
                    )

                    hypotheses = self._decode_batch(logits)
                    references = [batch['transcript'][i] for i in range(len(batch['transcript']))]

                    batch_wer = self.wer_calculator.compute_batch_wer(references, hypotheses)
                    batch_cer = self.cer_calculator.compute_batch_cer(references, hypotheses)

                    self.val_metrics.update(
                        loss.item(),
                        batch_wer['wer'],
                        batch_cer['cer'],
                        len(references)
                    )

                except RuntimeError as e:
                    print(f"Validation error: {e}")
                    continue

        val_metrics = self.val_metrics.compute()
        return val_metrics['loss'], val_metrics['wer']

    def test(self) -> Dict[str, float]:
        """在测试集上评估模型"""
        self.model.eval()
        test_metrics = RunningMetrics()

        all_hypotheses = []
        all_references = []

        with torch.no_grad():
            for batch in self.test_loader:
                audio_features = batch['audio_features'].to(self.config.DEVICE)
                video_features = batch['video_features'].to(self.config.DEVICE)
                targets = batch['targets'].to(self.config.DEVICE)
                target_lengths = batch['target_lengths'].to(self.config.DEVICE)
                audio_lengths = batch['audio_lengths'].to(self.config.DEVICE)
                video_lengths = batch['video_lengths'].to(self.config.DEVICE)

                logits = self.model(
                    audio_features, video_features,
                    audio_lengths, video_lengths
                )

                loss = self.criterion(
                    logits.transpose(0, 1),
                    targets,
                    torch.full((audio_features.size(0),), logits.size(1), dtype=torch.long, device=self.config.DEVICE),
                    target_lengths
                )

                hypotheses = self._decode_batch(logits)
                references = [batch['transcript'][i] for i in range(len(batch['transcript']))]

                batch_wer = self.wer_calculator.compute_batch_wer(references, hypotheses)
                batch_cer = self.cer_calculator.compute_batch_cer(references, hypotheses)

                test_metrics.update(
                    loss.item(),
                    batch_wer['wer'],
                    batch_cer['cer'],
                    len(references)
                )

                all_hypotheses.extend(hypotheses)
                all_references.extend(references)

        results = test_metrics.compute()
        results['samples'] = len(all_hypotheses)

        print(f"Test Results:")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  WER: {results['wer']*100:.2f}%")
        print(f"  CER: {results['cer']*100:.2f}%")

        return results

    def _decode_batch(self, logits: torch.Tensor) -> List[str]:
        """批量解码"""
        predictions = logits.argmax(dim=-1)
        hypotheses = []

        for pred in predictions:
            hyp = self.text_processor.decode(pred)
            hypotheses.append(hyp)

        return hypotheses

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_wer': self.best_wer,
            'patience_counter': self.patience_counter,
            'config': {
                'audio_encoder_channels': self.config.AUDIO_ENCODER_CHANNELS,
                'video_encoder_channels': self.config.VIDEO_ENCODER_CHANNELS,
                'audio_feature_dim': self.config.AUDIO_ENCODER_FEATURE_DIM,
                'video_feature_dim': self.config.VIDEO_ENCODER_FEATURE_DIM,
                'vocab_size': self.config.get_vocab_size()
            }
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"

        torch.save(checkpoint, path)
        return path

    def _save_final_checkpoint(self):
        """保存最终checkpoint"""
        checkpoint = {
            'epoch': self.config.MAX_EPOCHS,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_wer': self.best_wer,
            'config': {
                'audio_encoder_channels': self.config.AUDIO_ENCODER_CHANNELS,
                'video_encoder_channels': self.config.VIDEO_ENCODER_CHANNELS,
                'audio_feature_dim': self.config.AUDIO_ENCODER_FEATURE_DIM,
                'video_feature_dim': self.config.VIDEO_ENCODER_FEATURE_DIM,
                'vocab_size': self.config.get_vocab_size()
            }
        }

        path = self.checkpoint_dir / "final_model.pth"
        torch.save(checkpoint, path)
        print(f"Final model saved to {path}")

    def _save_training_history(self, history: Dict):
        """保存训练历史"""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def load_checkpoint(self, path: str) -> int:
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_wer = checkpoint.get('best_wer', float('inf'))

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from epoch {self.start_epoch}")

        return self.start_epoch

    def _log_training(self, epoch: int, train_loss: float, train_wer: float,
                     val_loss: float, val_wer: float):
        """记录训练日志"""
        log_entry = {
            'epoch': epoch,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': train_loss,
            'train_wer': train_wer,
            'val_loss': val_loss,
            'val_wer': val_wer,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        param_count = self.model.get_param_count()
        return {
            'total_params': param_count['total_params'],
            'trainable_params': param_count['trainable_params'],
            'vocab_size': self.config.get_vocab_size(),
            'device': self.config.DEVICE
        }


class DistributedTrainer:
    """分布式训练器（占位）"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Distributed training not implemented yet")
