# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

# 导入必要的库
import random
import argparse
from tqdm.auto import tqdm
import os
import torch
import wandb
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Callable
import shutil

# 导入自定义模块
from dataset import MSSDataset  # 音源分离数据集类
from utils import get_model_from_config  # 从配置文件获取模型
from valid import valid_multi_gpu, valid  # 验证函数

from utils import bind_lora_to_model, load_start_checkpoint
import loralib as lora  # LoRA低秩适应

import warnings

warnings.filterwarnings("ignore")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    解析命令行参数,用于配置模型、数据集和训练参数
    
    主要参数包括:
    - model_type: 选择使用的模型类型(mdx23c/htdemucs等)
    - config_path: 配置文件路径
    - data_path: 训练数据路径
    - dataset_type: 数据集类型(1-4)
    - device_ids: GPU设备ID
    - metrics: 评估指标列表
    - train_lora: 是否使用LoRA训练
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, edge_bs_rof, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--use_multistft_loss", action='store_true', help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def manual_seed(seed: int) -> None:
    """
    设置随机种子以确保实验可重复性
    
    包括:
    - Python random库
    - NumPy
    - PyTorch CPU和GPU
    - CUDA后端
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    初始化训练环境
    
    包括:
    - 设置随机种子
    - 配置PyTorch设置
    - 创建结果保存目录
    """
    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)

def wandb_init(args: argparse.Namespace, config: Dict, device_ids: List[int], batch_size: int) -> None:
    """
    初始化wandb日志系统
    
    用于:
    - 记录训练过程
    - 可视化训练指标
    - 保存实验配置
    """
    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(project='msst', config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size })


def prepare_data(config: Dict, args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    准备训练数据
    
    主要步骤:
    1. 创建MSSDataset实例
    2. 配置DataLoader参数
    3. 返回训练数据加载器
    """
    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader


def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    初始化模型并分配到合适的设备
    
    处理:
    1. 单GPU/多GPU配置
    2. CPU回退支持
    3. DataParallel并行处理
    """
    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    根据配置初始化优化器
    
    支持的优化器:
    - Adam: 自适应矩估计
    - AdamW: 权重衰减的Adam
    - RAdam: 修正的Adam
    - RMSprop: 均方根传播
    - Prodigy: 新型优化器
    - SGD: 随机梯度下降
    """
    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    """
    计算带掩码的损失函数
    
    实现:
    1. 计算MSE损失
    2. 基于分位数生成掩码
    3. 应用掩码得到最终损失
    
    形状:
    - y_: [音轨数, batch大小, 通道数, 音频长度]
    - y: 同y_
    """
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def multistft_loss(y: torch.Tensor, y_: torch.Tensor, loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    计算多分辨率STFT损失
    
    用于:
    - 频域和时域的联合优化
    - 改善音频重建质量
    
    支持:
    - 4D张量(标准模型)
    - 3D张量(Apollo等模型)
    """
    if len(y_.shape) == 4:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
    elif len(y_.shape) == 3:
        y1_ = y_
        y1 = y
    else:
        raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 3 or 4 dimensions.")

    return loss_multistft(y1_, y1)


def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    选择合适的损失函数
    
    支持的损失函数组合:
    1. MultiSTFT + MSE + L1
    2. MultiSTFT + MSE
    3. MultiSTFT + L1
    4. 仅MultiSTFT
    5. MSE + L1
    6. 仅MSE
    7. 仅L1
    8. Masked Loss
    """
    if args.use_multistft_loss:
        loss_options = dict(getattr(config, 'loss_multistft', {}))
        print(f'Loss options: {loss_options}')
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

        if args.use_mse_loss and args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        elif args.use_mse_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y)
        elif args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + F.l1_loss(y_, y)
        else:
            def multi_loss(y_, y):
                return multistft_loss(y_, y, loss_multistft) / 1000
    elif args.use_mse_loss:
        if args.use_l1_loss:
            def multi_loss(y_, y):
                return nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        else:
            multi_loss = nn.MSELoss()
    elif args.use_l1_loss:
        multi_loss = F.l1_loss
    else:
        def multi_loss(y_, y):
            return masked_loss(y_, y, q=config.training.q, coarse=config.training.coarse_loss_clip)
    return multi_loss


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对批次数据进行标准化
    
    步骤:
    1. 计算均值
    2. 计算标准差
    3. 应用标准化
    """
    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool, scaler: torch.cuda.amp.GradScaler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    """
    训练模型一个epoch
    
    主要步骤:
    1. 数据预处理和标准化
    2. 前向传播
    3. 损失计算
    4. 反向传播
    5. 梯度累积和优化器更新
    6. 指标记录
    
    特殊处理:
    - 自动混合精度训练
    - 梯度裁剪
    - 不同模型架构的特殊处理
    """
    model.train().to(device)
    print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
    loss_val = 0.
    total = 0

    normalize = getattr(config.training, 'normalize', False)

    pbar = tqdm(train_loader)
    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)  # mixture
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.model_type in ['mel_band_roformer', 'edge_bs_rof']:
                # loss is computed in forward pass
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    # If it's multiple GPUs sum partial loss
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y)

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()
        if config.training.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        li = loss.item() * gradient_accumulation_steps
        loss_val += li
        total += 1
        pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
        wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
        loss.detach()

    print(f'Training loss: {loss_val / total}')
    wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def save_weights(args: argparse.Namespace, store_path_prefix: str, model: torch.nn.Module, device_ids: List[int],
                 train_lora: bool, epoch: int, metric_value: float, is_early_stop: bool) -> None:
    """
    保存模型权重
    支持:
    - 普通模型权重保存
    - LoRA权重保存
    - 多GPU模型权重保存
    """
    if train_lora:
        suffix = f'_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_value:.4f}.ckpt'
        if is_early_stop:
            store_path = os.path.join(store_path_prefix, f'early_stop{suffix}')
        else:
            store_path = os.path.join(store_path_prefix, f'model{suffix}')
        torch.save(lora.lora_state_dict(model), store_path)
        best_model_path = os.path.join(store_path_prefix, "best_model.ckpt")
        shutil.copy(store_path, best_model_path)
    else:
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        suffix = f'_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_value:.4f}.ckpt'
        if is_early_stop:
            store_path = os.path.join(store_path_prefix, f'early_stop{suffix}')
        else:
            store_path = os.path.join(store_path_prefix, f'model{suffix}')
        torch.save(state_dict, store_path)
        best_model_path = os.path.join(store_path_prefix, "best_model.ckpt")
        shutil.copy(store_path, best_model_path)


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], epoch: int,
                          scheduler: torch.optim.lr_scheduler._LRScheduler, best_metric: float) -> Tuple[float, float]:
    """
    计算并记录当前epoch的评估指标
    
    主要功能:
    1. 验证模型性能
    2. 调整学习率
    3. 记录wandb日志
    """
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
    else:
        metrics_avg = valid(model, args, config, device, verbose=False)
    
    # 判断是否需要保存模型
    current_metric = metrics_avg[args.metric_for_scheduler]
    if current_metric > best_metric:
        best_metric = current_metric
        save_weights(args, args.results_path, model, device_ids, args.train_lora, epoch, current_metric, is_early_stop=False)
    
    scheduler.step(current_metric)
    wandb.log({'metric_main': current_metric})
    for metric_name in metrics_avg:
        wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})
    return current_metric, best_metric


def train_model(args: argparse.Namespace) -> None:
    """
    模型训练的主函数
    
    完整训练流程:
    1. 参数解析和环境初始化
    2. 模型和数据准备
    3. 优化器和损失函数配置
    4. 训练循环
       - 每个epoch的训练
       - 验证和指标计算
       - 模型保存
       - 学习率调整
    5. wandb日志记录
    
    支持特性:
    - 多GPU训练
    - 混合精度训练
    - LoRA微调
    - 断点续训
    - 多种评估指标
    """
    args = parse_args(args)

    initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    batch_size = config.training.batch_size * len(device_ids)

    wandb_init(args, config, device_ids, batch_size)

    train_loader = prepare_data(config, args, batch_size)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='train')

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    device, model = initialize_model_and_device(model, args.device_ids)

    if args.pre_valid:
        if torch.cuda.is_available() and len(device_ids) > 1:
            valid_multi_gpu(model, args, config, args.device_ids, verbose=True)
        else:
            valid(model, args, config, device, verbose=True)

    optimizer = get_optimizer(config, model)
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # Reduce LR if no metric improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                  factor=config.training.reduce_factor)

    multi_loss = choice_loss(args, config)
    scaler = GradScaler()

    # 读取早停配置
    early_stop = getattr(config.training, 'early_stop', {})
    early_stop_enabled = early_stop.get('enabled', False)
    early_stop_patience = early_stop.get('patience', 5)
    metric_for_early_stop = early_stop.get('metric', args.metric_for_scheduler)
    best_metric = float('-inf')
    no_improvement_count = 0

    print(
        f"Instruments: {config.training.instruments}\n"
        f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
        f"Patience: {config.training.patience} "
        f"Reduce factor: {config.training.reduce_factor}\n"
        f"Batch size: {batch_size} "
        f"Grad accum steps: {gradient_accumulation_steps} "
        f"Effective batch size: {batch_size * gradient_accumulation_steps}\n"
        f"Dataset type: {args.dataset_type}\n"
        f"Optimizer: {config.training.optimizer}"
    )

    print(f'Train for: {config.training.num_epochs} epochs')

    for epoch in range(config.training.num_epochs):
        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss)
        current_metric, best_metric = compute_epoch_metrics(model, args, config, device, device_ids, epoch, scheduler, best_metric)
        if early_stop_enabled:
            # 如果本 epoch 指标没有刷新 best_metric，则累计计数；否则重置连续无提升计数
            if current_metric < best_metric:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            # 如果连续无提升达到设定值，则提前停止训练
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping: {metric_for_early_stop} has not improved for {early_stop_patience} consecutive epochs.")
                save_weights(args, args.results_path, model, device_ids, args.train_lora, epoch, current_metric, is_early_stop=True)
                break

if __name__ == "__main__":
    train_model(None)   