import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Union
from configs import config
from models import AVASRModel
from data.audio_preprocessor import AudioPreprocessor
from data.video_preprocessor import VideoPreprocessor
from data.text_processor import TextProcessor
from decoding.ctc_decoder import CTCDecoder
from utils.common import set_seed, get_model_size, Timer


class AVASRInference:
    """AV-ASR推理引擎

    功能：
    - 单样本/批量推理
    - ONNX模型导出
    - CPU/GPU推理
    - 延迟测量
    """

    def __init__(self,
                model_path: str = None,
                config=None,
                device: str = None,
                use_onnx: bool = False,
                onnx_path: str = None):
        """
        Args:
            model_path: 模型权重路径
            config: 配置对象
            device: 推理设备 ("cuda" 或 "cpu")
            use_onnx: 是否使用ONNX模型
            onnx_path: ONNX模型路径
        """
        self.config = config or config
        self.device = device or self.config.DEVICE
        self.use_onnx = use_onnx

        self.audio_preprocessor = AudioPreprocessor(self.config)
        self.video_preprocessor = VideoPreprocessor(self.config)
        self.text_processor = TextProcessor(self.config)
        self.decoder = CTCDecoder(self.config)
        self.decoder.idx_to_char = self.text_processor.idx_to_char

        self.model = None
        self.onnx_session = None

        if not self.use_onnx:
            self.model = self._load_pytorch_model(model_path)
        else:
            self.onnx_session = self._load_onnx_model(onnx_path)

        self.warmup()

    def _load_pytorch_model(self, model_path: str) -> nn.Module:
        """加载PyTorch模型"""
        model = AVASRModel(self.config)
        model = model.to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model checkpoint provided. Using random weights.")

        model.eval()
        model_info = model.get_param_count()
        print(f"Model parameters: {model_info['total_params']:,}")
        print(f"Model size: {get_model_size(model):.2f} MB")

        return model

    def _load_onnx_model(self, onnx_path: str):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"Loaded ONNX model from {onnx_path}")
            return self.onnx_session
        except ImportError:
            print("onnxruntime not installed. Install with: pip install onnxruntime")
            return None

    def warmup(self):
        """预热推理引擎"""
        print("Warming up inference engine...")
        dummy_audio = torch.randn(1, 1, self.config.AUDIO_N_MELS, 500).to(self.device)
        dummy_video = torch.randn(1, 150, 1, 64, 64).to(self.device)

        with torch.no_grad():
            for _ in range(5):
                if not self.use_onnx:
                    _ = self.model(dummy_audio, dummy_video)

        print("Warmup completed.")

    def transcribe(self,
                  audio_path: str = None,
                  video_path: str = None,
                  audio_waveform: torch.Tensor = None,
                  video_frames: torch.Tensor = None,
                  transcript: str = None) -> Dict:
        """
        转录音频/视频

        Args:
            audio_path: 音频文件路径
            video_path: 视频文件路径
            audio_waveform: 音频波形张量
            video_frames: 视频帧张量
            transcript: 参考转录文本（用于WER计算，可选）

        Returns:
            result: 转录结果字典
        """
        start_time = time.time()

        if audio_waveform is None and audio_path:
            audio_waveform = self.audio_preprocessor.extract_features_from_file(audio_path)

        if video_frames is None and video_path:
            video_frames = self.video_preprocessor.extract_frames_from_video(video_path)

        if audio_waveform is None:
            raise ValueError("Either audio_path or audio_waveform must be provided")

        audio_features = self.audio_preprocessor(audio_waveform.to(self.device))

        if video_frames is not None:
            if video_frames.dim() == 3:
                video_frames = video_frames.unsqueeze(0)
            video_features = self.video_preprocessor(video_frames).to(self.device)
        else:
            batch_size = audio_features.shape[0]
            video_features = torch.randn(batch_size, 150, 1, 64, 64).to(self.device)

        with torch.no_grad():
            if not self.use_onnx:
                logits = self.model(audio_features, video_features)
            else:
                logits = self._onnx_inference(audio_features, video_features)

        hypothesis = self.decoder.greedy_decode(logits)
        decoded_text = self.text_processor.decode(hypothesis)

        elapsed_time = time.time() - start_time

        result = {
            'transcript': decoded_text,
            'confidence': self._compute_confidence(logits),
            'latency_ms': elapsed_time * 1000,
            'audio_length': audio_features.shape[-1] / 100,
            'num_frames': logits.shape[1]
        }

        if transcript:
            from data.text_processor import CharacterVocab
            wer = CharacterVocab.compute_wer(transcript, decoded_text)
            cer = CharacterVocab.compute_cer(transcript, decoded_text)
            result['reference'] = transcript
            result['wer'] = wer
            result['cer'] = cer

        return result

    def _onnx_inference(self, audio_features: torch.Tensor,
                       video_features: torch.Tensor) -> torch.Tensor:
        """ONNX模型推理"""
        import onnxruntime as ort

        audio_np = audio_features.cpu().numpy()
        video_np = video_features.cpu().numpy()

        input_names = [inp.name for inp in self.onnx_session.get_inputs()]

        if len(input_names) == 2:
            inputs = {
                input_names[0]: audio_np,
                input_names[1]: video_np
            }
        else:
            inputs = {input_names[0]: audio_np}

        outputs = self.onnx_session.run(None, inputs)
        return torch.from_numpy(outputs[0]).to(self.device)

    def _compute_confidence(self, logits: torch.Tensor) -> float:
        """计算置信度"""
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        return avg_confidence

    def batch_transcribe(self,
                        samples: List[Dict],
                        show_progress: bool = True) -> List[Dict]:
        """
        批量转录

        Args:
            samples: 样本列表，每个样本包含audio_path/video_path/transcript
            show_progress: 显示进度条

        Returns:
            results: 转录结果列表
        """
        results = []

        for i, sample in enumerate(samples):
            if show_progress:
                print(f"Processing sample {i+1}/{len(samples)}...")

            result = self.transcribe(
                audio_path=sample.get('audio_path'),
                video_path=sample.get('video_path'),
                transcript=sample.get('transcript')
            )
            results.append(result)

        return results

    def export_onnx(self,
                   output_path: str,
                   sample_audio: torch.Tensor = None,
                   sample_video: torch.Tensor = None):
        """导出ONNX模型"""
        if sample_audio is None:
            sample_audio = torch.randn(1, 1, self.config.AUDIO_N_MELS, 500)
        if sample_video is None:
            sample_video = torch.randn(1, 150, 1, 64, 64)

        from utils.common import export_to_onnx

        export_to_onnx(
            self.model,
            sample_audio.to(self.device),
            sample_video.to(self.device),
            output_path
        )

    def measure_latency(self,
                       num_samples: int = 100,
                       warmup: int = 10) -> Dict:
        """测量推理延迟"""
        print(f"Measuring latency with {num_samples} samples...")

        dummy_audio = torch.randn(1, 1, self.config.AUDIO_N_MELS, 500).to(self.device)
        dummy_video = torch.randn(1, 150, 1, 64, 64).to(self.device)

        latencies = []

        with torch.no_grad():
            for i in range(warmup + num_samples):
                start = time.time()
                if not self.use_onnx:
                    _ = self.model(dummy_audio, dummy_video)
                else:
                    _ = self._onnx_inference(dummy_audio, dummy_video)
                latency = time.time() - start

                if i >= warmup:
                    latencies.append(latency)

        latencies_ms = [l * 1000 for l in latencies]

        return {
            'mean_ms': sum(latencies_ms) / len(latencies_ms),
            'std_ms': (sum((x - sum(latencies_ms) / len(latencies_ms)) ** 2 for x in latencies_ms) / len(latencies_ms)) ** 0.5,
            'min_ms': min(latencies_ms),
            'max_ms': max(latencies_ms),
            'num_samples': num_samples,
            'device': self.device
        }

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.use_onnx:
            info = self.model.get_param_count()
            info['size_mb'] = get_model_size(self.model)
            info['device'] = self.device
            info['mode'] = 'pytorch'
        else:
            info = {
                'mode': 'onnx',
                'device': self.device
            }
        return info


class InferencePipeline:
    """推理流水线（简化版）"""

    def __init__(self, model_path: str = None, device: str = None):
        self.engine = AVASRInference(model_path, device=device)

    def run(self, audio_path: str, video_path: str = None) -> str:
        """运行推理"""
        result = self.engine.transcribe(audio_path, video_path)
        return result['transcript']

    def transcribe_file(self, audio_path: str) -> str:
        """转录音频文件"""
        return self.run(audio_path)


def create_inference_engine(model_path: str = None,
                           device: str = None) -> AVASRInference:
    """工厂函数：创建推理引擎"""
    return AVASRInference(model_path, device=device)
