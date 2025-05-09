# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

# 导入所需的库
import argparse
import time
import os
import glob
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union
from utils import demix, get_model_from_config, prefer_target_instrument, draw_spectrogram
from utils import normalize_audio, denormalize_audio, apply_tta, read_audio_transposed, load_start_checkpoint
from metrics import get_metrics
import warnings

warnings.filterwarnings("ignore")


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> None:
    """
    记录验证过程中的日志信息
    
    参数:
    ----------
    store_dir : str
        存储日志的目录,如果为空则不存储
    logs : List[str] 
        存储日志的列表
    text : str
        需要记录的文本信息
    """

    print(text)
    if verbose_logging:
        logs.append(text)


def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    将验证结果写入文件
    
    参数:
    ----------
    store_dir : str
        结果文件的存储目录
    results : List[str]
        需要写入文件的结果列表
    """
    with open(f'{store_dir}/results.txt', 'w') as out:
        for item in logs:
            out.write(item + "\n")


def get_mixture_paths(
    args,
    verbose: bool,
    config: ConfigDict,
    extension: str
) -> List[str]:
    """
    获取待验证的混合音频文件路径
    
    参数:
    ----------
    valid_path : List[str]
        验证数据集的目录列表
    verbose : bool
        是否打印详细信息
    config : ConfigDict
        配置对象,包含推理参数如重叠数和批大小
    extension : str
        音频文件扩展名
        
    返回:
    -------
    List[str]
        混合音频文件路径列表
    """
    try:
        valid_path = args.valid_path
    except Exception as e:
        print('No valid path in args')
        raise e

    all_mixtures_path = []
    for path in valid_path:
        part = sorted(glob.glob(f"{path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                print(f'No validation data found in: {path}')
        all_mixtures_path += part
    if verbose:
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')

    return all_mixtures_path


def update_metrics_and_pbar(
        track_metrics: Dict,
        all_metrics: Dict,
        instr: str,
        pbar_dict: Dict,
        mixture_paths: Union[List[str], tqdm],
        verbose: bool = False
) -> None:
    """
    更新评估指标和进度条
    
    参数:
    ----------
    track_metrics : Dict
        当前音轨的评估指标字典
    all_metrics : Dict
        所有音轨的评估指标汇总字典
    instr : str
        当前处理的乐器名称
    pbar_dict : Dict
        进度条显示的指标字典
    mixture_paths : tqdm
        进度条对象
    verbose : bool
        是否打印详细信息
    """
    for metric_name, metric_value in track_metrics.items():
        if verbose:
            print(f"Metric {metric_name:11s} value: {metric_value:.4f}")
        all_metrics[metric_name][instr].append(metric_value)
        pbar_dict[f'{metric_name}_{instr}'] = metric_value

    if mixture_paths is not None:
        try:
            mixture_paths.set_postfix(pbar_dict)
        except Exception:
            pass


def process_audio_files(
    mixture_paths: List[str],
    model: torch.nn.Module,
    args,
    config,
    device: torch.device,
    verbose: bool = False,
    is_tqdm: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    处理音频文件并进行源分离评估
    
    参数:
    ----------
    mixture_paths : List[str]
        混合音频文件路径列表
    model : torch.nn.Module
        训练好的源分离模型
    args : Any
        包含用户指定选项的参数对象
    config : Any
        包含模型和处理参数的配置对象
    device : torch.device
        运行设备(CPU或CUDA)
    verbose : bool
        是否打印详细日志
    is_tqdm : bool
        是否显示进度条
        
    返回:
    -------
    Dict[str, Dict[str, List[float]]]
        评估指标的嵌套字典,外层key为指标名称,内层key为乐器名称
    """
    # 获取目标乐器列表
    instruments = prefer_target_instrument(config)

    # 获取测试时增强(TTA)设置
    use_tta = getattr(args, 'use_tta', False)
    # 获取文件存储目录
    store_dir = getattr(args, 'store_dir', '')
    # 获取音频编码格式
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # 初始化评估指标字典
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    if is_tqdm:
        mixture_paths = tqdm(mixture_paths)

    # 遍历处理每个混合音频文件
    for path in mixture_paths:
        start_time = time.time()
        # 读取混合音频
        mix, sr = read_audio_transposed(path)
        mix_orig = mix.copy()
        folder = os.path.dirname(path)

        # 重采样到目标采样率
        if 'sample_rate' in config.audio:
            if sr != config.audio['sample_rate']:
                orig_length = mix.shape[-1]
                if verbose:
                    print(f'Warning: sample rate is different. In config: {config.audio["sample_rate"]} in file {path}: {sr}')
                mix = librosa.resample(mix, orig_sr=sr, target_sr=config.audio['sample_rate'], res_type='kaiser_best')

        if verbose:
            folder_name = os.path.abspath(folder)
            print(f'Song: {folder_name} Shape: {mix.shape}')

        # 音频归一化
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        # 使用模型进行源分离
        waveforms_orig = demix(config, model, mix.copy(), device, model_type=args.model_type)

        # 应用测试时增强
        if use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        pbar_dict = {}

        # 对每个乐器分别计算评估指标
        for instr in instruments:
            if verbose:
                print(f"Instr: {instr}")

            # 读取原始乐器音轨作为参考
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = read_audio_transposed(f"{folder}/{instr}.{extension}", instr, skip_err=True)
                if track is None:
                    continue
            else:
                # 如果是other轨道,需要从vocals轨道计算
                track, sr1 = read_audio_transposed(f"{folder}/vocals.{extension}")
                track = mix_orig - track

            estimates = waveforms_orig[instr]

            # 重采样到原始采样率
            if 'sample_rate' in config.audio:
                if sr != config.audio['sample_rate']:
                    estimates = librosa.resample(estimates, orig_sr=config.audio['sample_rate'], target_sr=sr,
                                                 res_type='kaiser_best')
                    estimates = librosa.util.fix_length(estimates, size=orig_length)

            # 反归一化
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            # 保存分离结果
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.wav"
                sf.write(out_wav_name, estimates.T, sr, subtype='FLOAT')
                if args.draw_spectro > 0:
                    out_img_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.jpg"
                    draw_spectrogram(estimates.T, sr, args.draw_spectro, out_img_name)
                    out_img_name_orig = f"{store_dir}/{os.path.basename(folder)}_{instr}_orig.jpg"
                    draw_spectrogram(track.T, sr, args.draw_spectro, out_img_name_orig)

            # 计算评估指标
            track_metrics = get_metrics(
                args.metrics,
                track,
                estimates,
                mix_orig,
                device=device,
            )

            # 更新评估指标和进度条
            update_metrics_and_pbar(
                track_metrics,
                all_metrics,
                instr, pbar_dict,
                mixture_paths=mixture_paths,
                verbose=verbose
            )

        if verbose:
            print(f"Time for song: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: List[str],
    config: ConfigDict,
    all_metrics: Dict[str, Dict[str, List[float]]],
    start_time: float
) -> Dict[str, float]:
    """
    计算并记录每个乐器的平均评估指标
    
    参数:
    ----------
    store_dir : str
        日志存储目录
    args : dict
        参数字典
    instruments : List[str]
        乐器列表
    config : ConfigDict
        配置字典
    all_metrics : Dict[str, Dict[str, List[float]]]
        所有评估指标的字典
    start_time : float
        开始时间
        
    返回:
    -------
    Dict[str, float]
        所有乐器的平均评估指标
    """

    logs = []
    if store_dir:
        logs.append(str(args))
        verbose_logging = True
    else:
        verbose_logging = False

    logging(logs, text=f"Num overlap: {config.inference.num_overlap}", verbose_logging=verbose_logging)

    metric_avg = {}
    # 计算每个乐器的评估指标均值和标准差
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])

            mean_val = metric_values.mean()
            std_val = metric_values.std()

            logging(logs, text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})", verbose_logging=verbose_logging)
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    
    # 计算所有乐器的平均指标
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            logging(logs, text=f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}', verbose_logging=verbose_logging)
    logging(logs, text=f"Elapsed time: {time.time() - start_time:.2f} sec", verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False
) -> dict:
    """
    在单个设备上验证模型
    
    参数:
    ----------
    model : torch.nn.Module
        源分离模型
    args : Namespace
        命令行参数
    config : dict
        配置字典
    device : torch.device
        运行设备
    verbose : bool
        是否打印详细信息
        
    返回:
    -------
    dict
        所有乐器的平均评估指标
    """

    start_time = time.time()
    model.eval().to(device)

    # 获取存储目录
    store_dir = getattr(args, 'store_dir', '')
    # 获取音频编码格式
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # 获取所有混合音频文件路径
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    # 处理音频文件并计算评估指标
    all_metrics = process_audio_files(all_mixtures_path, model, args, config, device, verbose, not verbose)
    instruments = prefer_target_instrument(config)

    # 计算平均评估指标
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_mixtures_path: List[str],
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: str,
    return_dict
) -> None:
    """
    在子进程中执行验证,支持多进程并行处理
    
    参数:
    ----------
    proc_id : int
        进程ID
    queue : torch.multiprocessing.Queue
        用于接收混合音频文件路径的队列
    all_mixtures_path : List[str]
        所有混合音频文件路径
    model : torch.nn.Module
        源分离模型
    args : dict
        参数字典
    config : ConfigDict
        配置对象
    device : str
        运行设备
    return_dict : torch.multiprocessing.Manager().dict
        用于存储每个进程结果的共享字典
    """

    m1 = model.eval().to(device)
    if proc_id == 0:
        progress_bar = tqdm(total=len(all_mixtures_path))

    # 初始化评估指标字典
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    while True:
        current_step, path = queue.get()
        if path is None:  # 检查结束标记
            break
        single_metrics = process_audio_files([path], m1, args, config, device, False, False)
        pbar_dict = {}
        for instr in config.training.instruments:
            for metric_name in all_metrics:
                all_metrics[metric_name][instr] += single_metrics[metric_name][instr]
                if len(single_metrics[metric_name][instr]) > 0:
                    pbar_dict[f"{metric_name}_{instr}"] = f"{single_metrics[metric_name][instr][0]:.4f}"
        if proc_id == 0:
            progress_bar.update(current_step - progress_bar.n)
            progress_bar.set_postfix(pbar_dict)
    return_dict[proc_id] = all_metrics
    return


def run_parallel_validation(
    verbose: bool,
    all_mixtures_path: List[str],
    config: ConfigDict,
    model: torch.nn.Module,
    device_ids: List[int],
    args,
    return_dict
) -> None:
    """
    运行多进程并行验证
    
    参数:
    ----------
    verbose : bool
        是否打印详细信息
    all_mixtures_path : List[str]
        所有混合音频文件路径
    config : ConfigDict
        配置对象
    model : torch.nn.Module
        源分离模型
    device_ids : List[int]
        GPU设备ID列表
    args : dict
        参数字典
    return_dict
        用于存储所有进程结果的共享字典
    """

    model = model.to('cpu')
    try:
        # 对于多GPU训练提取单个模型
        model = model.module
    except:
        pass

    queue = torch.multiprocessing.Queue()
    processes = []

    # 为每个设备创建一个进程
    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_mixtures_path, model, args, config, device, return_dict)
        )
        p.start()
        processes.append(p)
    
    # 向队列中添加任务
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    # 添加结束标记
    for _ in range(len(device_ids)):
        queue.put((None, None))
    # 等待所有进程完成
    for p in processes:
        p.join()

    return


def valid_multi_gpu(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device_ids: List[int],
    verbose: bool = False
) -> Dict[str, float]:
    """
    在多个GPU上执行验证
    
    参数:
    ----------
    model : torch.nn.Module
        源分离模型
    args : dict
        参数字典
    config : ConfigDict
        配置对象
    device_ids : List[int]
        GPU设备ID列表
    verbose : bool
        是否打印详细信息
        
    返回:
    -------
    Dict[str, float]
        每个评估指标的平均值
    """

    start_time = time.time()

    # 获取存储目录
    store_dir = getattr(args, 'store_dir', '')
    # 获取音频编码格式
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # 获取所有混合音频文件路径
    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    # 创建共享字典存储结果
    return_dict = torch.multiprocessing.Manager().dict()

    # 运行并行验证
    run_parallel_validation(verbose, all_mixtures_path, config, model, device_ids, args, return_dict)

    # 合并所有进程的结果
    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []
            for i in range(len(device_ids)):
                all_metrics[metric][instr] += return_dict[i][metric][instr]

    instruments = prefer_target_instrument(config)

    # 计算平均评估指标
    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time)


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    解析命令行参数
    
    参数:
    ----------
    dict_args: Dict
        命令行参数字典,如果为None则从sys.argv解析
        
    返回:
    -------
    argparse.Namespace
        解析后的参数对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
                             " edge_bs_rof, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint"
                                                                          " to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='List of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def check_validation(dict_args):
    """
    执行验证的主函数
    
    参数:
    ----------
    dict_args
        命令行参数字典
    """
    args = parse_args(dict_args)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    
    # 获取模型和配置
    model, config = get_model_from_config(args.model_type, args.config_path)

    # 加载检查点
    if args.start_check_point:
        load_start_checkpoint(args, model, type_='valid')

    print(f"Instruments: {config.training.instruments}")

    # 设置运行设备
    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')

    # 根据设备数量选择验证方式
    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
