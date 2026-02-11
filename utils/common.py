import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional
from configs import config


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """获取模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """估算模型FLOPs"""
    from torch.profiler import profile, record_function

    def count_fn(m, inputs, outputs):
        if isinstance(m, nn.Conv2d):
            c_in = m.in_channels
            c_out = m.out_channels
            k_h, k_w = m.kernel_size
            output_h, output_w = outputs[0].shape[2:]
            flops = c_in * c_out * k_h * k_w * output_h * output_w
        elif isinstance(m, nn.Linear):
            flops = m.in_features * m.out_features
        else:
            flops = 0
        m.__flops__ += flops

    model.__flops__ = 0
    for m in model.modules():
        m.register_forward_hook(count_fn)

    with torch.no_grad():
        model(input_tensor)

    return model.__flops__


def init_logger(log_file: str = None):
    """初始化日志记录"""
    import logging

    logger = logging.getLogger('AV-ASR')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """平均计量器"""

    def __init__(self, name: str = "Meter"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class ProgressMeter:
    """进度显示器"""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmt = '{desc}[{1:{len}d}/{2:{len}d}] {3}'
        self.num_batches = num_batches
        self.meters = meters
        self.prefix = prefix
        self.len = len(str(num_batches))

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmt.format(batch, batch, self.num_batches, str(meter)) for meter in self.meters]
        print('\r' + ' '.join(entries), end='\r')


def export_to_onnx(model: nn.Module,
                  input_sample: torch.Tensor,
                  video_sample: torch.Tensor,
                  output_path: str,
                  opset_version: int = 11):
    """导出模型到ONNX格式

    Args:
        model: PyTorch模型
        input_sample: 音频输入样例 [1, 1, F, T]
        video_sample: 视频输入样例 [1, T_v, C, H, W]
        output_path: 输出路径
        opset_version: ONNX opset版本
    """
    model.eval()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        (input_sample, video_sample),
        output_path,
        input_names=['audio_features', 'video_features'],
        output_names=['logits'],
        dynamic_axes={
            'audio_features': {3: 'audio_length'},
            'video_features': {1: 'video_length'},
            'logits': {1: 'output_length'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    print(f"Model exported to {output_path}")

    verify_onnx_model(output_path)

    return output_path


def verify_onnx_model(model_path: str):
    """验证ONNX模型"""
    try:
        import onnx
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("ONNX model verification passed!")
        return True
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
        return False


def load_onnx_model(model_path: str):
    """加载ONNX模型进行推理"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        return session
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        return None


def onnx_inference(session,
                  audio_features: np.ndarray,
                  video_features: np.ndarray):
    """ONNX模型推理"""
    input_names = [inp.name for inp in session.get_inputs()]

    if len(input_names) == 2:
        inputs = {
            input_names[0]: audio_features,
            input_names[1]: video_features
        }
    else:
        inputs = {input_names[0]: audio_features}

    outputs = session.run(None, inputs)
    return outputs[0]


def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'device': torch.cuda.get_device_name(0)
        }
    return None


def print_model_summary(model: nn.Module,
                       input_sizes: list,
                       device: str = "cuda"):
    """打印模型摘要"""
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print("-" * 80)

    print("\nLayers:")
    layer_count = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_count += 1

    print(f"Total layers: {layer_count}")

    print("=" * 80 + "\n")


class Timer:
    """计时器"""

    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def start(self):
        self.start_time = time.time()
        self.elapsed_time = None

    def stop(self):
        if self.start_time is not None:
            self.elapsed_time = time.time() - self.start_time
            return self.elapsed_time
        return 0

    def reset(self):
        self.start_time = None
        self.elapsed_time = None

    @property
    def value(self):
        if self.elapsed_time is not None:
            return self.elapsed_time
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"
