"""
创建测试混合音频的脚本

将两个音频文件A和B混合成16kHz的混合音频，用于测试语音分离模型

使用方法:
    python create_test_mix.py --audio_a speaker1.wav --audio_b speaker2.wav --output mixed.wav

    # 调整混合比例
    python create_test_mix.py --audio_a a.wav --audio_b b.wav --output mixed.wav --ratio_a 0.6 --ratio_b 0.4
"""

import argparse
import numpy as np
import soundfile as sf
import librosa


def load_and_resample(audio_path, target_sr=16000):
    """
    加载音频并重采样到目标采样率

    参数:
        audio_path: 音频文件路径
        target_sr: 目标采样率，默认16000

    返回:
        audio: 重采样后的音频数据
        sr: 目标采样率
    """
    print(f"加载音频: {audio_path}")

    # 读取音频
    audio, sr = sf.read(audio_path)

    print(f"  原始采样率: {sr} Hz")
    print(f"  原始长度: {len(audio)/sr:.2f} 秒")

    # 转换为单声道
    if audio.ndim > 1:
        print(f"  原始声道数: {audio.shape[1]}，转换为单声道")
        audio = audio.mean(axis=1)

    # 重采样到16kHz
    if sr != target_sr:
        print(f"  重采样: {sr} Hz -> {target_sr} Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio, target_sr


def align_audio_length(audio_a, audio_b, mode='pad'):
    """
    对齐两个音频的长度

    参数:
        audio_a: 音频A
        audio_b: 音频B
        mode: 对齐模式
            - 'pad': 短的用零填充（默认）
            - 'trim': 长的截断到短的长度
            - 'loop': 短的循环填充到长的长度

    返回:
        audio_a_aligned: 对齐后的音频A
        audio_b_aligned: 对齐后的音频B
    """
    len_a = len(audio_a)
    len_b = len(audio_b)

    print(f"\n对齐音频长度:")
    print(f"  音频A长度: {len_a} 采样点")
    print(f"  音频B长度: {len_b} 采样点")

    if len_a == len_b:
        print(f"  长度已对齐")
        return audio_a, audio_b

    if mode == 'pad':
        # 零填充模式
        max_len = max(len_a, len_b)
        if len_a < max_len:
            audio_a = np.pad(audio_a, (0, max_len - len_a), mode='constant')
            print(f"  音频A填充到 {max_len} 采样点")
        if len_b < max_len:
            audio_b = np.pad(audio_b, (0, max_len - len_b), mode='constant')
            print(f"  音频B填充到 {max_len} 采样点")

    elif mode == 'trim':
        # 截断模式
        min_len = min(len_a, len_b)
        audio_a = audio_a[:min_len]
        audio_b = audio_b[:min_len]
        print(f"  两个音频截断到 {min_len} 采样点")

    elif mode == 'loop':
        # 循环填充模式
        max_len = max(len_a, len_b)
        if len_a < max_len:
            repeats = int(np.ceil(max_len / len_a))
            audio_a = np.tile(audio_a, repeats)[:max_len]
            print(f"  音频A循环填充到 {max_len} 采样点")
        if len_b < max_len:
            repeats = int(np.ceil(max_len / len_b))
            audio_b = np.tile(audio_b, repeats)[:max_len]
            print(f"  音频B循环填充到 {max_len} 采样点")

    return audio_a, audio_b


def mix_audio(audio_a, audio_b, ratio_a=0.5, ratio_b=0.5, normalize=True):
    """
    混合两个音频

    参数:
        audio_a: 音频A
        audio_b: 音频B
        ratio_a: 音频A的混合比例（0-1）
        ratio_b: 音频B的混合比例（0-1）
        normalize: 是否归一化输出

    返回:
        mixed: 混合后的音频
    """
    print(f"\n混合音频:")
    print(f"  音频A比例: {ratio_a}")
    print(f"  音频B比例: {ratio_b}")

    # 混合
    mixed = ratio_a * audio_a + ratio_b * audio_b

    # 归一化
    if normalize:
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95
            print(f"  归一化: 最大值 {max_val:.3f} -> 0.95")
        elif max_val > 0:
            mixed = mixed / max_val * 0.95
            print(f"  归一化到 0.95")

    return mixed


def save_audio(audio, output_path, sr=16000):
    """
    保存音频文件

    参数:
        audio: 音频数据
        output_path: 输出文件路径
        sr: 采样率
    """
    sf.write(output_path, audio, sr)
    duration = len(audio) / sr
    print(f"\n保存混合音频:")
    print(f"  文件: {output_path}")
    print(f"  采样率: {sr} Hz")
    print(f"  长度: {duration:.2f} 秒")
    print(f"  采样点数: {len(audio)}")


def create_test_mix(audio_a_path, audio_b_path, output_path,
                   ratio_a=0.5, ratio_b=0.5,
                   align_mode='pad', target_sr=16000,
                   save_sources=False):
    """
    创建测试混合音频

    参数:
        audio_a_path: 音频A的路径
        audio_b_path: 音频B的路径
        output_path: 输出混合音频的路径
        ratio_a: 音频A的混合比例
        ratio_b: 音频B的混合比例
        align_mode: 对齐模式 ('pad', 'trim', 'loop')
        target_sr: 目标采样率
        save_sources: 是否保存处理后的源音频
    """
    print("=" * 60)
    print("创建测试混合音频")
    print("=" * 60)

    # 1. 加载并重采样音频
    audio_a, sr = load_and_resample(audio_a_path, target_sr)
    audio_b, sr = load_and_resample(audio_b_path, target_sr)

    # 2. 对齐长度
    audio_a, audio_b = align_audio_length(audio_a, audio_b, mode=align_mode)

    # 3. 混合音频
    mixed = mix_audio(audio_a, audio_b, ratio_a, ratio_b, normalize=True)

    # 4. 保存混合音频
    save_audio(mixed, output_path, sr)

    # 5. 可选：保存处理后的源音频
    if save_sources:
        import os
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        source_a_path = os.path.join(base_dir, f"{base_name}_source_a.wav")
        source_b_path = os.path.join(base_dir, f"{base_name}_source_b.wav")

        # 归一化源音频
        audio_a_norm = audio_a / np.abs(audio_a).max() * 0.95 if np.abs(audio_a).max() > 0 else audio_a
        audio_b_norm = audio_b / np.abs(audio_b).max() * 0.95 if np.abs(audio_b).max() > 0 else audio_b

        sf.write(source_a_path, audio_a_norm, sr)
        sf.write(source_b_path, audio_b_norm, sr)

        print(f"\n保存源音频:")
        print(f"  源音频A: {source_a_path}")
        print(f"  源音频B: {source_b_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='创建测试混合音频',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument('--audio_a', type=str, required=True,
                        help='音频A的路径')
    parser.add_argument('--audio_b', type=str, required=True,
                        help='音频B的路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出混合音频的路径')

    # 可选参数
    parser.add_argument('--ratio_a', type=float, default=0.5,
                        help='音频A的混合比例 (0-1)')
    parser.add_argument('--ratio_b', type=float, default=0.5,
                        help='音频B的混合比例 (0-1)')
    parser.add_argument('--align_mode', type=str, default='pad',
                        choices=['pad', 'trim', 'loop'],
                        help='长度对齐模式')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='目标采样率')
    parser.add_argument('--save_sources', action='store_true',
                        help='保存处理后的源音频（用于对比）')

    args = parser.parse_args()

    # 创建混合音频
    create_test_mix(
        audio_a_path=args.audio_a,
        audio_b_path=args.audio_b,
        output_path=args.output,
        ratio_a=args.ratio_a,
        ratio_b=args.ratio_b,
        align_mode=args.align_mode,
        target_sr=args.target_sr,
        save_sources=args.save_sources
    )


if __name__ == '__main__':
    main()
