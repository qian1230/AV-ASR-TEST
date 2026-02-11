import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


class CTCLoss(nn.Module):
    """CTC损失函数封装

    支持：
    - 标准CTC损失
    - 标签平滑CTC损失（迭代优化添加）
    """

    def __init__(self, config=None, reduction: str = "mean", blank_id: int = 0):
        super().__init__()
        self.config = config or config
        self.reduction = reduction
        self.blank_id = blank_id
        self.label_smoothing = self.config.LABEL_SMOOTHING

        self._ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction=self.reduction,
            zero_infinity=True
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        计算CTC损失

        Args:
            logits: 模型输出的logits [T, B, V] 或 [B, T, V]
            targets: 目标序列 [B, L]
            input_lengths: 输入序列长度 [B]
            target_lengths: 目标序列长度 [B]

        Returns:
            loss: CTC损失值
        """
        if logits.dim() == 3:
            logits = logits.transpose(0, 1)

        logits = F.log_softmax(logits, dim=-1)

        loss = self._ctc_loss(logits, targets, input_lengths, target_lengths)

        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return loss


class LabelSmoothingCTCLoss(nn.Module):
    """带标签平滑的CTC损失"""

    def __init__(self, config=None, reduction: str = "mean", blank_id: int = 0):
        super().__init__()
        self.config = config or config
        self.reduction = reduction
        self.blank_id = blank_id
        self.epsilon = self.config.LABEL_SMOOTHING
        self.vocab_size = self.config.get_vocab_size()

        self._ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction=self.reduction,
            zero_infinity=True
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> dict:
        """
        计算带标签平滑的CTC损失

        Args:
            logits: 模型输出的logits [T, B, V] 或 [B, T, V]
            targets: 目标序列 [B, L]
            input_lengths: 输入序列长度 [B]
            target_lengths: 目标序列长度 [B]

        Returns:
            dict: 包含总损失和kl散度的字典
        """
        if logits.dim() == 3:
            logits = logits.transpose(0, 1)

        log_probs = F.log_softmax(logits, dim=-1)

        ctc_loss = self._ctc_loss(log_probs, targets, input_lengths, target_lengths)

        if torch.isinf(ctc_loss) or torch.isnan(ctc_loss):
            ctc_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        kl_loss = self._compute_label_smoothing_kl(log_probs, input_lengths)

        total_loss = ctc_loss + 0.1 * kl_loss

        return {
            'total_loss': total_loss,
            'ctc_loss': ctc_loss,
            'kl_loss': kl_loss
        }

    def _compute_label_smoothing_kl(self, log_probs: torch.Tensor,
                                    input_lengths: torch.Tensor) -> torch.Tensor:
        """计算标签平滑的KL散度"""
        probs = log_probs.exp()

        uniform_dist = torch.ones_like(probs) / probs.size(-1)
        uniform_dist[:, :, self.blank_id] = 0

        smooth_targets = (1 - self.epsilon) * probs + self.epsilon * uniform_dist
        smooth_targets = smooth_targets / smooth_targets.sum(dim=-1, keepdim=True)

        kl_div = F.kl_div(
            log_probs,
            smooth_targets.detach(),
            reduction='none',
            log_target=True
        )

        mask = torch.zeros_like(kl_div)
        for b, length in enumerate(input_lengths):
            mask[b, :length, :] = 1

        masked_kl = kl_div * mask
        kl_loss = masked_kl.sum() / mask.sum()

        return kl_loss


class FocalCTCLoss(nn.Module):
    """Focal CTC损失（实验性）"""

    def __init__(self, config=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.config = config or config
        self.gamma = gamma
        self.reduction = reduction
        self.blank_id = self.config.BLANK_ID

        self._ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction=self.reduction,
            zero_infinity=True
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """计算Focal CTC损失"""
        if logits.dim() == 3:
            logits = logits.transpose(0, 1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        ctc_loss = self._ctc_loss(log_probs, targets, input_lengths, target_lengths)

        focal_weight = (1 - probs.mean(dim=-1)) ** self.gamma
        focal_loss = ctc_loss * focal_weight.mean()

        return focal_loss


class MixedCTCLoss(nn.Module):
    """混合CTC损失（多任务学习）"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or config

        self.ctc = CTCLoss(config)
        self.label_smooth_ctc = LabelSmoothingCTCLoss(config)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor,
                use_label_smoothing: bool = False) -> torch.Tensor:
        """计算混合CTC损失"""
        if use_label_smoothing:
            losses = self.label_smooth_ctc(logits, targets, input_lengths, target_lengths)
            return losses['total_loss']
        else:
            return self.ctc(logits, targets, input_lengths, target_lengths)


def create_loss_function(config, loss_type: str = "ctc") -> nn.Module:
    """工厂函数：创建损失函数"""
    if loss_type == "ctc":
        return CTCLoss(config)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCTCLoss(config)
    elif loss_type == "focal":
        return FocalCTCLoss(config)
    elif loss_type == "mixed":
        return MixedCTCLoss(config)
    else:
        return CTCLoss(config)
