# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

# 导入必要的库
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import soundfile as sf
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union
import loralib as lora


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    根据模型类型从指定路径加载配置文件
    
    参数:
    ----------
    model_type : str
        模型类型 (如 'htdemucs', 'mdx23c' 等)
    config_path : str
        YAML或OmegaConf配置文件的路径
        
    返回:
    -------
    config : Any
        加载的配置对象,可能是OmegaConf或ConfigDict格式
        
    异常:
    ------
    FileNotFoundError: 配置文件不存在时抛出
    ValueError: 加载配置文件出错时抛出
    """
    try:
        with open(config_path, 'r') as f:
            # htdemucs模型使用OmegaConf格式配置
            if model_type == 'htdemucs':
                config = OmegaConf.load(config_path)
            # 其他模型使用yaml格式配置
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    """
    根据模型类型和配置文件加载对应的模型
    
    参数:
    ----------
    model_type : str
        模型类型 (如 'mdx23c', 'htdemucs', 'scnet' 等)
    config_path : str
        配置文件路径(YAML或OmegaConf格式)
        
    返回:
    -------
    model : nn.Module or None
        根据model_type初始化的模型实例
    config : Any
        用于初始化模型的配置对象
        
    异常:
    ------
    ValueError: 未知的model_type或模型初始化错误时抛出
    """

    config = load_config(model_type, config_path)

    # 根据不同的模型类型加载对应的模型架构
    if model_type == 'mdx23c':
        # MDX23C模型 - 使用TFC-TDF网络架构
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        # HTDemucs模型 - 基于Demucs的高质量音源分离模型
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        # 分割模型 - 用于音频分割任务
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        # TorchSeg模型 - PyTorch实现的分割模型
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        # 基于Mel频带的Roformer模型
        from models.edge_bs_rof import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'edge_bs_rof':
        # 基础Roformer模型
        from models.edge_bs_rof import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'swin_upernet':
        # Swin Transformer + UperNet架构
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        # Bandit模型 - 多掩码多源带分离RNN
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        # Bandit V2模型 - 改进版本
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        # 非官方SCNet实现
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        # 官方SCNet实现
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'apollo':
        # Apollo模型 - Look2Hear框架中的模型
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'bs_mamba2':
        # BS-Mamba2模型 - 基于Mamba架构的分离器
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        # 实验性MDX23C模型 - 带STHT的TFC-TDF网络
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'dcunet':
        # DCUNet模型
        from models.dcunet import DCUNet
        model = DCUNet(config)
    elif model_type == 'dprnn':
        # DPRNN模型 - 基于深度循环神经网络的音源分离模型
        from models.dprnn.dprnn import DPRNN
        model = DPRNN(config)
    elif model_type == 'dptnet':
        # DPTNet模型 - 基于双路径变换网络的音源分离模型
        from models.dptnet.dpt_net import DPTNet
        model = DPTNet(config)
        

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def read_audio_transposed(path: str, instr: str = None, skip_err: bool = False) -> Tuple[np.ndarray, int]:
    """
    读取音频文件并进行转置处理
    
    参数:
    ----------
    path : str
        音频文件路径
    skip_err: bool
        是否跳过错误
    instr: str
        乐器名称
        
    返回:
    -------
    Tuple[np.ndarray, int]
        - 转置后的音频数据,形状为(channels, length)
        - 采样率(如44100)
    """

    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            print(f"No stem {instr}: skip!")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        # 单声道音频转为二维数组
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    对音频信号进行归一化处理
    
    参数:
    ----------
    audio : np.ndarray
        输入音频数组,形状为(channels, time)或(time,)
        
    返回:
    -------
    tuple[np.ndarray, dict[str, float]]
        - 归一化后的音频数组
        - 包含均值和标准差的字典
    """

    # 计算单声道信号
    mono = audio.mean(0)
    # 计算均值和标准差
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    对归一化的音频信号进行反归一化
    
    参数:
    ----------
    audio : np.ndarray
        归一化后的音频数组
    norm_params : dict[str, float]
        包含均值和标准差的字典
        
    返回:
    -------
    np.ndarray
        反归一化后的音频数组
    """

    return audio * norm_params["std"] + norm_params["mean"]


def apply_tta(
        config,
        model: torch.nn.Module,
        mix: torch.Tensor,
        waveforms_orig: Dict[str, torch.Tensor],
        device: torch.device,
        model_type: str
) -> Dict[str, torch.Tensor]:
    """
    应用测试时数据增强(TTA)进行音源分离
    
    通过对输入混音进行通道反转和极性反转等增强,
    然后对所有增强结果取平均来提高分离效果
    
    参数:
    ----------
    config : Any
        模型配置对象
    model : torch.nn.Module
        训练好的模型
    mix : torch.Tensor
        混音音频张量(channels, time)
    waveforms_orig : Dict[str, torch.Tensor]
        原始分离波形字典
    device : torch.device
        运行设备(CPU/CUDA)
    model_type : str
        模型类型
        
    返回:
    -------
    Dict[str, torch.Tensor]
        应用TTA后更新的分离波形字典
    """
    # 创建增强:通道反转和极性反转
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # 处理每个增强后的混音
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # 对所有增强结果取平均
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    生成带有线性淡入淡出的窗口数组
    
    参数:
    ----------
    window_size : int
        窗口总大小
    fade_size : int
        淡入淡出区域的大小
        
    返回:
    -------
    torch.Tensor
        生成的窗口数组,形状为(window_size,)
    """

    # 生成淡入淡出序列
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    # 创建窗口并应用淡入淡出
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    统一的音源分离函数,支持多种处理模式
    
    使用重叠窗口分块处理的方式进行高效无伪影的分离
    
    参数:
    ----------
    config : ConfigDict
        音频和推理设置配置
    model : torch.nn.Module
        训练好的分离模型
    mix : torch.Tensor
        输入混音张量(channels, time)
    device : torch.device
        计算设备
    model_type : str
        模型类型,如"demucs"等
    pbar : bool
        是否显示进度条
        
    返回:
    -------
    Union[Dict[str, np.ndarray], np.ndarray]
        - 多乐器时返回乐器到分离音频的映射字典
        - 单乐器时返回分离的音频数组
    """

    mix = torch.tensor(mix, dtype=torch.float32)

    # 根据模型类型选择处理模式
    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
        
    # 根据模式设置处理参数
    if mode == 'demucs':
        # Demucs模式参数
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        # 通用模式参数
        chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # 添加边界填充
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # 初始化结果和计数器张量
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False
            ) if pbar else None

            # 分块处理音频
            while i < mix.shape[1]:
                # 提取并填充音频块
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # 处理满足batch_size的批次
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone()
                        if i - step == 0:  # 第一块不需要淡入
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # 最后一块不需要淡出
                            window[-fade_size:] = 1

                    # 将处理结果加入总结果
                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # 计算最终估计源
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # 移除通用模式的填充
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # 返回结果
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
    根据配置返回目标乐器列表
    
    参数:
    ----------
    config : ConfigDict
        包含乐器列表或目标乐器的配置对象
        
    返回:
    -------
    List[str]
        目标乐器列表
    """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments


def load_not_compatible_weights(model: torch.nn.Module, weights: str, verbose: bool = False) -> None:
    """
    加载不完全兼容的权重到模型中
    
    参数:
    ----------
    model: 目标PyTorch模型
    weights: 权重文件路径
    verbose: 是否打印详细信息
    """

    new_model = model.state_dict()
    old_model = torch.load(weights)
    if 'state' in old_model:
        # htdemucs权重加载修复
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # apollo权重加载修复
        old_model = old_model['state_dict']

    # 遍历新模型的每一层
    for el in new_model:
        if el in old_model:
            if verbose:
                print(f'Match found for {el}!')
            if new_model[el].shape == old_model[el].shape:
                # 形状相同直接复制
                if verbose:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                # 处理形状不同的情况
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if verbose:
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
                    # 处理不同形状的权重
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if verbose:
                print(f'Match not found for {el}!')
    model.load_state_dict(
        new_model
    )


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    加载LoRA权重到模型中
    
    参数:
    ----------
    model : Module
        目标PyTorch模型
    lora_path : str
        LoRA检查点文件路径
    device : str
        加载权重的设备
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)


def load_start_checkpoint(args: argparse.Namespace, model: torch.nn.Module, type_='train') -> None:
    """
    加载模型的起始检查点
    
    参数:
    ----------
    args: 包含检查点路径的命令行参数
    model: 要加载检查点的PyTorch模型
    type_: 加载权重的方式
    """

    print(f'Start from checkpoint: {args.start_check_point}')
    if type_ in ['train']:
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))
    else:
        device='cpu'
        if args.model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=False)
            # htdemucs预训练模型修复
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # apollo预训练模型修复
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    if args.lora_checkpoint:
        print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        load_lora_weights(model, args.lora_checkpoint)


def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    """
    将模型中的特定层替换为LoRA扩展版本
    
    参数:
    ----------
    config : Dict[str, Any]
        包含LoRA参数的配置
    model : nn.Module
        要替换层的原始模型
        
    返回:
    -------
    nn.Module
        替换层后的模型
    """

    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    replaced_layers = 0  # 替换层计数器

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]

        # 检查是否为目标替换层
        if isinstance(module, nn.Linear):
            try:
                # 获取父模块
                parent_module = model
                for submodule_name in hierarchy[:-1]:
                    parent_module = getattr(parent_module, submodule_name)

                # 用LoRA层替换原始层
                setattr(
                    parent_module,
                    layer_name,
                    lora.MergedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        **config['lora']
                    )
                )
                replaced_layers += 1

            except Exception as e:
                print(f"Error replacing layer {name}: {e}")

    if replaced_layers == 0:
        print("Warning: No layers were replaced. Check the model structure and configuration.")
    else:
        print(f"Number of layers replaced with LoRA: {replaced_layers}")

    return model


def draw_spectrogram(waveform, sample_rate, length, output_file):
    """
    绘制音频波形的频谱图
    
    参数:
    ----------
    waveform: 音频波形数据
    sample_rate: 采样率
    length: 绘制长度
    output_file: 输出文件路径
    """
    import librosa.display

    # 截取所需部分的频谱图
    x = waveform[:int(length * sample_rate), :]
    # 对单声道信号进行短时傅里叶变换
    X = librosa.stft(x.mean(axis=-1))
    # 将幅度谱转换为dB刻度的频谱图
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    fig, ax = plt.subplots()
    # 显示频谱图
    img = librosa.display.specshow(
        Xdb,
        cmap='plasma',
        sr=sample_rate,
        x_axis='time',
        y_axis='linear',
        ax=ax
    )
    ax.set(title='File: ' + os.path.basename(output_file))
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    if output_file is not None:
        plt.savefig(output_file)
