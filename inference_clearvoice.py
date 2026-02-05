"""
基于ClearVoice代码风格的MossFormer2_SS_16K推理脚本

参考：https://github.com/modelscope/ClearerVoice-Studio

使用方法:
    python inference_clearvoice_style.py --input_file mixed.wav --output_dir ./output
"""

import os
import argparse
import torch
import numpy as np
import soundfile as sf
from argparse import Namespace
from mossformer2 import MossFormer2_SS_16K


def decode_one_audio_mossformer2_ss_16k(model, device, inputs, sampling_rate=16000,
                                        decode_window=10.0, one_time_decode_length=30,
                                        num_spks=2):
    """
    使用MossFormer2模型进行16kHz语音分离的解码函数

    该函数支持分段处理长音频，使用滑动窗口和重叠拼接来避免边界伪影。

    参数:
        model: MossFormer2模型
        device: 运行设备 ('cuda' 或 'cpu')
        inputs: 输入音频numpy数组，形状为 (batch, time)
        sampling_rate: 采样率，默认16000
        decode_window: 解码窗口长度（秒），默认10秒
        one_time_decode_length: 一次性解码的最大长度（秒），默认30秒
        num_spks: 说话人数量，默认2

    返回:
        list: 包含各个说话人分离后音频的列表
    """
    out = []
    decode_do_segment = False
    window = int(sampling_rate * decode_window)  # 解码窗口长度
    stride = int(window * 0.75)  # 解码步长（75%重叠）
    b, t = inputs.shape

    # 计算输入的RMS值用于后续归一化
    rms_input = (inputs ** 2).mean() ** 0.5

    # 判断是否需要分段处理
    if t > sampling_rate * one_time_decode_length:
        decode_do_segment = True

    # 对输入进行填充以满足窗口要求
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], axis=1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)

    # 转换为torch张量
    inputs = torch.from_numpy(np.float32(inputs)).to(device)
    b, t = inputs.shape

    if decode_do_segment:
        # 分段处理模式
        print(f"  使用分段处理模式（音频较长）")
        outputs = np.zeros((num_spks, t))
        give_up_length = (window - stride) // 2  # 每段舍弃的长度
        current_idx = 0

        segment_count = 0
        while current_idx + window <= t:
            segment_count += 1
            tmp_input = inputs[:, current_idx:current_idx + window]

            # 模型前向传播
            tmp_out_list = model(tmp_input)

            for spk in range(num_spks):
                tmp_out_list[spk] = tmp_out_list[spk][0, :].detach().cpu().numpy()
                if current_idx == 0:
                    # 第一段：保留开头到(window - give_up_length)
                    outputs[spk, current_idx:current_idx + window - give_up_length] = \
                        tmp_out_list[spk][:-give_up_length]
                else:
                    # 后续段：舍弃两端的give_up_length
                    outputs[spk, current_idx + give_up_length:current_idx + window - give_up_length] = \
                        tmp_out_list[spk][give_up_length:-give_up_length]

            current_idx += stride

        print(f"  处理了 {segment_count} 个音频段")
        for spk in range(num_spks):
            out.append(outputs[spk, :])
    else:
        # 一次性处理模式
        print(f"  使用一次性处理模式")
        out_list = model(inputs)
        for spk in range(num_spks):
            out.append(out_list[spk][0, :].detach().cpu().numpy())

    # RMS归一化：将输出归一化到输入的能量水平
    for spk in range(num_spks):
        rms_out = (out[spk] ** 2).mean() ** 0.5
        if rms_out > 0:
            out[spk] = out[spk] / rms_out * rms_input

    return out


def inference_single_file(model, device, input_file, output_dir, args):
    """
    对单个音频文件进行语音分离推理

    参数:
        model: MossFormer2模型
        device: 运行设备
        input_file: 输入音频文件路径
        output_dir: 输出目录
        args: 配置参数
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取音频文件
    print(f"\n正在加载音频: {input_file}")
    audio, sr = sf.read(input_file)

    # 处理采样率
    if sr != args.sampling_rate:
        print(f"警告: 音频采样率为 {sr} Hz，需要重采样到 {args.sampling_rate} Hz")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sampling_rate)

    # 处理声道
    if audio.ndim > 1:
        print(f"警告: 音频是 {audio.shape[1]} 声道，转换为单声道")
        audio = audio.mean(axis=1)

    # 转换为批次格式 [B, T]
    audio = audio.reshape(1, -1)

    print(f"音频长度: {audio.shape[1]/args.sampling_rate:.2f} 秒")
    print("正在进行语音分离...")

    # 执行推理
    with torch.no_grad():
        separated_audios = decode_one_audio_mossformer2_ss_16k(
            model=model.model,  # MossFormer2_SS_16K.model 是实际的MossFormer模型
            device=device,
            inputs=audio,
            sampling_rate=args.sampling_rate,
            decode_window=args.decode_window,
            one_time_decode_length=args.one_time_decode_length,
            num_spks=args.num_spks
        )

    # 保存分离后的音频
    print(f"\n保存分离后的音频到: {output_dir}")
    basename = os.path.splitext(os.path.basename(input_file))[0]

    for spk_idx, audio_data in enumerate(separated_audios):
        # 归一化到[-0.95, 0.95]防止削波
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95

        output_file = os.path.join(output_dir, f'{basename}_s{spk_idx+1}.wav')
        sf.write(output_file, audio_data, args.sampling_rate)
        print(f"  - 说话人 {spk_idx+1}: {output_file}")

    print("\n语音分离完成!")


def main():
    parser = argparse.ArgumentParser(
        description='MossFormer2 16K语音分离推理 (ClearVoice风格)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入输出参数
    parser.add_argument('--input_file', type=str, required=True,
                        help='输入混合音频文件路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./model',
                        help='checkpoint目录')

    # 模型参数
    parser.add_argument('--num_spks', type=int, default=2,
                        help='说话人数量')
    parser.add_argument('--encoder_kernel_size', type=int, default=16,
                        help='编码器卷积核大小')
    parser.add_argument('--encoder_embedding_dim', type=int, default=512,
                        help='编码器嵌入维度')
    parser.add_argument('--mossformer_sequence_dim', type=int, default=512,
                        help='MossFormer序列维度')
    parser.add_argument('--num_mossformer_layer', type=int, default=24,
                        help='MossFormer层数')

    # 推理参数
    parser.add_argument('--sampling_rate', type=int, default=16000,
                        help='采样率')
    parser.add_argument('--one_time_decode_length', type=float, default=30.0,
                        help='一次性解码的最大长度（秒）')
    parser.add_argument('--decode_window', type=float, default=10.0,
                        help='解码窗口长度（秒）')
    parser.add_argument('--use_cuda', type=int, default=1,
                        help='使用CUDA (1=是, 0=否)')

    args = parser.parse_args()

    # 设置设备
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    # 检查输入文件
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    # 检查checkpoint
    checkpoint_file = os.path.join(args.checkpoint_dir, 'last_best_checkpoint.pt')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint文件不存在: {checkpoint_file}\n"
            f"请从 https://huggingface.co/alibabasglab/MossFormer2_SS_16K 下载模型"
        )

    # 初始化模型
    print("\n初始化MossFormer2_SS_16K模型...")
    model = MossFormer2_SS_16K(args)

    # 加载checkpoint
    print(f"加载checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # 处理不同的checkpoint格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 加载权重（处理可能的module.前缀）
    model_state = model.model.state_dict()
    for key in model_state.keys():
        if key in state_dict and model_state[key].shape == state_dict[key].shape:
            model_state[key] = state_dict[key]
        elif key.replace('module.', '') in state_dict:
            model_state[key] = state_dict[key.replace('module.', '')]
        elif 'module.' + key in state_dict:
            model_state[key] = state_dict['module.' + key]

    model.model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print("模型加载完成!")

    # 执行推理
    inference_single_file(model, device, args.input_file, args.output_dir, args)


if __name__ == '__main__':
    main()
