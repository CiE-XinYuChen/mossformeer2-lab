"""
MossFormer2 16K音频语音分离推理脚本

使用方法:
    python inference_16k.py --input_file mixed_audio.wav --output_dir ./output

Author: Based on MossFormer2 architecture
"""

import os
import argparse
import torch
import torchaudio
from argparse import Namespace


def load_model(checkpoint_path, device='cuda'):
    """
    加载MossFormer2_SS_16K模型

    Args:
        checkpoint_path: checkpoint文件路径
        device: 运行设备 ('cuda' 或 'cpu')

    Returns:
        model: 加载好的模型
    """
    # 导入模型定义
    from mossformer2 import MossFormer2_SS_16K

    # 模型配置参数（16K版本）
    args = Namespace(
        encoder_embedding_dim=512,      # 编码器嵌入维度
        mossformer_sequence_dim=512,    # MossFormer序列维度
        num_mossformer_layer=24,        # MossFormer层数
        encoder_kernel_size=16,         # 编码器卷积核大小
        num_spks=2,                     # 分离的说话人数量
    )

    # 初始化模型
    model = MossFormer2_SS_16K(args)

    # 加载checkpoint
    print(f"正在从 {checkpoint_path} 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 根据checkpoint结构加载权重
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"模型已加载到 {device}")
    return model, args.num_spks


def inference(model, input_file, output_dir, num_spks=2, sample_rate=16000):
    """
    对单个音频文件进行语音分离推理

    Args:
        model: MossFormer2_SS_16K模型
        input_file: 输入混合音频文件路径
        output_dir: 输出目录
        num_spks: 说话人数量
        sample_rate: 采样率（16000 Hz）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载音频文件
    print(f"\n正在加载音频文件: {input_file}")
    waveform, sr = torchaudio.load(input_file)

    # 检查采样率
    if sr != sample_rate:
        print(f"警告: 音频采样率为 {sr} Hz, 需要重采样到 {sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # 如果是立体声，转换为单声道
    if waveform.shape[0] > 1:
        print(f"警告: 音频是 {waveform.shape[0]} 声道，转换为单声道")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    print(f"音频长度: {waveform.shape[1]/sample_rate:.2f} 秒")

    # 移动到模型所在设备
    device = next(model.parameters()).device
    waveform = waveform.to(device)

    # 推理
    print("正在进行语音分离...")
    with torch.no_grad():
        # 模型输入: [B, T], 输出: list of [B, T]
        separated_sources = model(waveform)

    # 保存分离后的音频
    print(f"\n保存分离后的音频到: {output_dir}")
    for i, source in enumerate(separated_sources):
        # 归一化防止削波
        source_cpu = source.detach().cpu()
        max_val = torch.abs(source_cpu).max()
        if max_val > 0:
            source_cpu = source_cpu / max_val * 0.95

        output_file = os.path.join(output_dir, f'speaker_{i+1}.wav')
        torchaudio.save(output_file, source_cpu, sample_rate)
        print(f"  - 说话人 {i+1}: {output_file}")

    print("\n语音分离完成!")


def main():
    parser = argparse.ArgumentParser(description='MossFormer2 16K语音分离推理')
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入混合音频文件路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录 (默认: ./output)')
    parser.add_argument('--checkpoint', type=str,
                        default='./model/last_best_checkpoint.pt',
                        help='模型checkpoint路径 (默认: ./model/last_best_checkpoint.pt)')
    parser.add_argument('--num_spks', type=int, default=2,
                        help='说话人数量 (默认: 2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda 或 cpu, 默认: cuda)')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'

    # 检查输入文件
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    # 检查checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint文件不存在: {args.checkpoint}")

    # 加载模型
    model, num_spks = load_model(args.checkpoint, args.device)

    # 推理
    inference(model, args.input_file, args.output_dir, num_spks)


if __name__ == '__main__':
    main()
