"""
AV-ASR 项目验证脚本

功能：
1. 验证项目文件结构完整性
2. 检查代码语法正确性
3. 验证导入依赖
4. 验证模型创建
"""
import os
import sys
import subprocess
from pathlib import Path


class ProjectValidator:
    """项目验证器"""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.required_files = [
            'configs/__init__.py',
            'configs/config.py',
            'data/__init__.py',
            'data/audio_preprocessor.py',
            'data/video_preprocessor.py',
            'data/text_processor.py',
            'data/dataset.py',
            'models/__init__.py',
            'models/audio_encoder.py',
            'models/video_encoder.py',
            'models/fusion.py',
            'models/av_asr_model.py',
            'training/__init__.py',
            'training/trainer.py',
            'training/loss.py',
            'training/metrics.py',
            'decoding/__init__.py',
            'decoding/ctc_decoder.py',
            'utils/__init__.py',
            'utils/common.py',
            'scripts/train.py',
            'scripts/test.py',
            'scripts/inference.py',
            'requirements.txt',
            'README.md'
        ]
        self.errors = []
        self.warnings = []
        self.success = []

    def validate_structure(self):
        """验证项目结构"""
        print("=" * 60)
        print("验证项目文件结构")
        print("=" * 60)

        for file_path in self.required_files:
            full_path = self.project_dir / file_path
            if full_path.exists():
                print(f"✓ {file_path}")
                self.success.append(file_path)
            else:
                print(f"✗ {file_path} - 文件不存在!")
                self.errors.append(f"Missing file: {file_path}")

        print(f"\n总计: {len(self.success)}/{len(self.required_files)} 文件存在")

    def validate_python_syntax(self):
        """验证Python语法"""
        print("\n" + "=" * 60)
        print("验证Python语法")
        print("=" * 60)

        python_files = list(self.project_dir.rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, py_file, 'exec')
                print(f"✓ {py_file.relative_to(self.project_dir)}")
            except SyntaxError as e:
                print(f"✗ {py_file.relative_to(self.project_dir)} - 语法错误: {e}")
                self.errors.append(f"Syntax error in {py_file}: {e}")

        print(f"\n检查了 {len(python_files)} 个Python文件")

    def validate_imports(self):
        """验证导入"""
        print("\n" + "=" * 60)
        print("验证Python导入")
        print("=" * 60)

        modules_to_test = [
            ('configs.config', '配置模块'),
            ('data.audio_preprocessor', '音频预处理'),
            ('data.video_preprocessor', '视频预处理'),
            ('data.text_processor', '文本处理'),
            ('data.dataset', '数据集'),
            ('models.audio_encoder', '音频编码器'),
            ('models.video_encoder', '视频编码器'),
            ('models.fusion', '特征融合'),
            ('models.av_asr_model', '完整模型'),
            ('training.trainer', '训练器'),
            ('training.loss', 'CTC损失'),
            ('training.metrics', '评估指标'),
            ('decoding.ctc_decoder', 'CTC解码器'),
            ('utils.common', '通用工具'),
        ]

        for module_name, description in modules_to_test:
            try:
                module = __import__(module_name, fromlist=[''])
                print(f"✓ {module_name} ({description})")
            except ImportError as e:
                print(f"✗ {module_name} - 导入错误: {e}")
                self.warnings.append(f"Import error in {module_name}: {e}")

    def validate_model_creation(self):
        """验证模型创建"""
        print("\n" + "=" * 60)
        print("验证模型创建")
        print("=" * 60)

        try:
            import torch

            from configs import config
            from models import AVASRModel

            print("创建模型...")
            model = AVASRModel(config)

            param_count = model.get_param_count()
            print(f"✓ 模型创建成功")
            print(f"  - 总参数量: {param_count['total_params']:,}")
            print(f"  - 可训练参数量: {param_count['trainable_params']:,}")

            if param_count['total_params'] > 100_000_000:
                self.warnings.append(f"参数量超过100M限制: {param_count['total_params']:,}")

        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            self.errors.append(f"Model creation failed: {e}")

    def validate_forward_pass(self):
        """验证前向传播"""
        print("\n" + "=" * 60)
        print("验证前向传播")
        print("=" * 60)

        try:
            import torch
            from configs import config
            from models import AVASRModel

            model = AVASRModel(config)
            model.eval()

            B, T_audio = 2, 500
            B, T_video = 2, 150

            audio_input = torch.randn(B, 1, config.AUDIO_N_MELS, T_audio)
            video_input = torch.randn(B, T_video, 1, 64, 64)

            with torch.no_grad():
                logits = model(audio_input, video_input)

            print(f"✓ 前向传播成功")
            print(f"  - 输入形状: 音频={audio_input.shape}, 视频={video_input.shape}")
            print(f"  - 输出形状: {logits.shape}")
            print(f"  - 词汇表大小: {config.get_vocab_size()}")

        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            self.errors.append(f"Forward pass failed: {e}")

    def run_all(self):
        """运行所有验证"""
        print("\n" + "=" * 60)
        print("AV-ASR 项目验证")
        print("=" * 60)

        self.validate_structure()
        self.validate_python_syntax()
        self.validate_imports()

        print("\n" + "=" * 60)
        print("模型功能验证")
        print("=" * 60)
        self.validate_model_creation()
        self.validate_forward_pass()

        self.print_summary()

        return len(self.errors) == 0

    def print_summary(self):
        """打印验证总结"""
        print("\n" + "=" * 60)
        print("验证总结")
        print("=" * 60)

        print(f"✓ 成功项: {len(self.success)}")
        print(f"✗ 错误项: {len(self.errors)}")
        print(f"⚠ 警告项: {len(self.warnings)}")

        if self.errors:
            print("\n错误详情:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n警告详情:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if len(self.errors) == 0:
            print("\n🎉 项目验证通过！")
        else:
            print("\n⚠ 项目验证发现问题，请检查错误。")


def check_dependencies():
    """检查依赖安装"""
    print("=" * 60)
    print("检查依赖")
    print("=" * 60)

    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'opencv-python': 'OpenCV',
    }

    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {name} ({package})")
        except ImportError:
            print(f"✗ {name} ({package}) - 未安装")
            all_installed = False

    if not all_installed:
        print("\n请安装缺失的依赖:")
        print("pip install -r requirements.txt")

    return all_installed


def main():
    """主函数"""
    project_dir = Path(__file__).parent

    print("\n" + "=" * 60)
    print("AV-ASR 项目验证工具")
    print("=" * 60)
    print(f"项目路径: {project_dir}")

    check_dependencies()

    validator = ProjectValidator(project_dir)
    success = validator.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
