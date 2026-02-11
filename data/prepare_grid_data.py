"""
GRID数据集预处理脚本

功能：
1. 解析alignment JSON文件，提取转录文本
2. 生成训练/验证/测试清单文件
3. 处理数据划分（按说话人）
"""
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

PHONEME_TO_CHAR = {
    # 元音
    'aa': 'a', 'ae': 'a', 'ah': 'a', 'ao': 'a', 'aw': 'a',
    'ay': 'a', 'ea': 'e', 'eh': 'e', 'er': 'r', 'ey': 'e',
    'ih': 'i', 'iy': 'i', 'oa': 'o', 'oi': 'o', 'oo': 'o',
    'ow': 'o', 'oy': 'o', 'uh': 'u', 'uw': 'u', 'uy': 'u',
    'ax': '', 'ix': 'i', 'axr': 'r',
    # 辅音
    'b': 'b', 'ch': 't', 'd': 'd', 'dh': 'd', 'f': 'f',
    'g': 'g', 'hh': 'h', 'jh': 'j', 'k': 'k', 'l': 'l',
    'm': 'm', 'n': 'n', 'ng': 'n', 'p': 'p', 'r': 'r',
    's': 's', 'sh': 's', 't': 't', 'th': 't', 'v': 'v',
    'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'z',
    # 边界标记
    'B': '', 'I': '', 'E': '', 'S': '',
    # 静音
    'SIL': ' ', 'SIL_S': ' ', 'sp': ' ',
}

COMMANDS = [
    'bin', 'lay', 'place', 'set',
    'blue', 'green', 'red', 'white',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'zero',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'now', 'please', 'soon'
]

def extract_transcript_from_alignment(alignment_path: str) -> str:
    """从alignment JSON提取转录文本"""
    PHONEME_TO_CHAR = {
        'aa': 'a', 'ae': 'a', 'ah': 'a', 'ao': 'a', 'aw': 'a',
        'ay': 'a', 'ea': 'e', 'eh': 'e', 'ey': 'e',
        'ih': 'i', 'iy': 'i', 'oa': 'o', 'oy': 'o',
        'uh': 'u', 'uw': 'u',
        'b': 'b', 'ch': 't', 'd': 'd', 'dh': 'd', 'f': 'f',
        'g': 'g', 'hh': 'h', 'jh': 'j', 'k': 'k', 'l': 'l',
        'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r',
        's': 's', 'sh': 's', 't': 't', 'th': 't', 'v': 'v',
        'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'z', 'ng': 'n',
        'SIL': ' ', 'sp': ' ',
    }

    try:
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_chars = []

        for key in data:
            phonemes = data[key]

            for phoneme_info in phonemes:
                phone = phoneme_info.get('phone', '')

                parts = phone.split('_')
                if len(parts) == 2:
                    base_phone = parts[0]
                else:
                    base_phone = phone

                if base_phone in PHONEME_TO_CHAR:
                    char = PHONEME_TO_CHAR[base_phone]
                    all_chars.append(char)
                elif base_phone == 'er':
                    all_chars.append('r')

        transcript = ''.join(all_chars)

        while '  ' in transcript:
            transcript = transcript.replace('  ', ' ')
        transcript = transcript.strip()

        if not transcript:
            return None

        return transcript.lower()

    except Exception as e:
        print(f"Error processing {alignment_path}: {e}")
        return None


def extract_sample_id(filename: str) -> str:
    """从文件名提取样本ID"""
    base = Path(filename).stem
    return base


def check_files_exist(base_dir: Path, sample_id: str) -> Tuple[bool, bool]:
    """检查音频和视频文件是否存在"""
    audio_path = base_dir / 'audio' / f"{sample_id}.wav"
    video_path = base_dir / 'front' / f"{sample_id}.mov"

    audio_exists = os.path.exists(str(audio_path))
    video_exists = os.path.exists(str(video_path))

    return audio_exists, video_exists


def process_dataset(raw_data_dir: str, output_dir: str,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   seed: int = 42) -> Dict:
    """
    处理GRID数据集

    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        处理统计信息
    """
    random.seed(seed)

    raw_dir = Path(raw_data_dir)
    out_dir = Path(output_dir)

    manifests_dir = out_dir / 'manifests'
    manifests_dir.mkdir(parents=True, exist_ok=True)

    speakers = ['s2', 's3']
    all_samples = []

    print("=" * 60)
    print("处理GRID数据集")
    print("=" * 60)

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for speaker in speakers:
        speaker_dir = raw_dir / speaker

        if not speaker_dir.exists():
            print(f"⚠ 说话人目录不存在: {speaker_dir}")
            continue

        alignment_dir = speaker_dir / 'alignment'

        if not alignment_dir.exists():
            print(f"⚠ Alignment目录不存在: {alignment_dir}")
            continue

        print(f"\n处理说话人: {speaker}")
        speaker_count = 0

        for alignment_file in alignment_dir.glob('*.json'):
            sample_id = extract_sample_id(alignment_file.name)

            audio_exists, video_exists = check_files_exist(speaker_dir, sample_id)

            if not audio_exists:
                total_skipped += 1
                continue

            transcript = extract_transcript_from_alignment(str(alignment_file))

            if transcript is None or len(transcript.strip()) == 0:
                total_errors += 1
                continue

            sample_info = {
                'sample_id': sample_id,
                'speaker': speaker,
                'audio_path': f"{speaker}/audio/{sample_id}.wav",
                'video_path': f"{speaker}/front/{sample_id}.mov",
                'alignment_path': f"{speaker}/alignment/{alignment_file.name}",
                'transcript': transcript
            }

            all_samples.append(sample_info)
            speaker_count += 1
            total_processed += 1

        print(f"  处理了 {speaker_count} 个样本")

    if not all_samples:
        print("❌ 未找到有效样本！")
        return {'error': 'No valid samples found'}

    random.shuffle(all_samples)

    n_samples = len(all_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]

    print(f"\n数据划分:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")

    def write_manifest(samples: List[Dict], filepath: Path):
        """写入清单文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                line = f"{sample['audio_path']}|{sample['video_path']}|{sample['transcript']}"
                f.write(line + '\n')

    write_manifest(train_samples, manifests_dir / 'train_manifest.txt')
    write_manifest(val_samples, manifests_dir / 'val_manifest.txt')
    write_manifest(test_samples, manifests_dir / 'test_manifest.txt')

    stats = {
        'total_samples': n_samples,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'processed': total_processed,
        'skipped': total_skipped,
        'errors': total_errors
    }

    print(f"\n清单文件已保存到: {manifests_dir}")
    print(f"  - {manifests_dir}/train_manifest.txt")
    print(f"  - {manifests_dir}/val_manifest.txt")
    print(f"  - {manifests_dir}/test_manifest.txt")

    return stats


def generate_vocab_from_samples(samples: List[Dict], output_path: str):
    """从样本生成词汇表"""
    all_chars = set()

    for sample in samples:
        for char in sample['transcript']:
            all_chars.add(char)

    vocab = sorted(list(all_chars))
    vocab = ['<blank>', ' '] + vocab

    print(f"生成的词汇表 ({len(vocab)} 个token):")
    print(vocab)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

    print(f"词汇表已保存到: {output_path}")

    return vocab


def analyze_transcripts(samples: List[Dict]):
    """分析转录文本"""
    print("\n" + "=" * 60)
    print("转录文本分析")
    print("=" * 60)

    lengths = [len(s['transcript']) for s in samples]

    print(f"样本数: {len(samples)}")
    print(f"平均长度: {sum(lengths)/len(lengths):.1f} 字符")
    print(f"最短: {min(lengths)} 字符")
    print(f"最长: {max(lengths)} 字符")

    vocab = set()
    for s in samples:
        for char in s['transcript']:
            vocab.add(char)

    print(f"唯一字符数: {len(vocab)}")
    print(f"字符集: {sorted(vocab)}")

    all_transcripts = [s['transcript'] for s in samples]
    unique_transcripts = set(all_transcripts)
    print(f"唯一转录数: {len(unique_transcripts)}")


def main():
    """主函数"""
    raw_data_dir = "D:/TRAE/Project/AV-ASR/data/raw"
    output_dir = "D:/TRAE/Project/AV-ASR/data"

    print("=" * 60)
    print("GRID数据集预处理")
    print("=" * 60)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出目录: {output_dir}")

    stats = process_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )

    if 'error' not in stats:
        manifests_dir = Path(output_dir) / 'manifests'

        train_samples = []
        with open(manifests_dir / 'train_manifest.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    train_samples.append({
                        'audio_path': parts[0],
                        'video_path': parts[1],
                        'transcript': parts[2]
                    })

        analyze_transcripts(train_samples)

        vocab_path = Path(output_dir) / 'vocabulary.json'
        generate_vocab_from_samples(train_samples, str(vocab_path))

        print("\n✅ 数据预处理完成！")
        print(f"处理统计: {stats}")
    else:
        print("❌ 数据预处理失败！")


if __name__ == "__main__":
    main()
