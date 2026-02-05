
"""
简单示例：使用MossFormer2_SS_16K进行语音分离

运行方式:
    python main.py
"""

import torch
import torchaudio
from argparse import Namespace
from mossformer2 import MossFormer2_SS_16K


if __name__ == '__main__':
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 配置模型参数
    args = Namespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )

    # 初始化模型
    print("初始化模型...")
    model = MossFormer2_SS_16K(args)

    # 加载checkpoint
    checkpoint_path = './model/last_best_checkpoint.pt'
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("模型加载完成!")

    # 示例：处理音频文件
    # 取消下面的注释并替换为你的音频文件路径
    """
    input_file = 'your_mixed_audio.wav'

    # 加载音频
    waveform, sr = torchaudio.load(input_file)

    # 重采样到16kHz（如果需要）
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # 转换为单声道（如果需要）
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.to(device)

    # 推理
    print("进行语音分离...")
    with torch.no_grad():
        separated_sources = model(waveform)

    # 保存结果
    for i, source in enumerate(separated_sources):
        source_cpu = source.detach().cpu()
        source_cpu = source_cpu / source_cpu.abs().max() * 0.95
        output_file = f'speaker_{i+1}.wav'
        torchaudio.save(output_file, source_cpu, 16000)
        print(f"保存: {output_file}")

    print("完成!")
    """

    print("\n提示: 编辑 main.py 中的代码，取消注释并设置你的音频文件路径")
    print("或者使用: python inference_16k.py --input_file your_audio.wav --output_dir ./output")
