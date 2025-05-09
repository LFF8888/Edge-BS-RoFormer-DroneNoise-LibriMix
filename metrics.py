# 导入必要的库
import numpy as np
import torch
import librosa
import torch.nn.functional as F
from typing import Dict, List, Tuple

def sdr(references: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """
    计算信号失真比(Signal-to-Distortion Ratio, SDR)
    
    SDR用于衡量预测源与参考源的匹配程度。它通过计算参考信号能量与误差能量(参考与预测的差异)的比值来评估分离质量。
    返回分贝(dB)单位的SDR值。
    
    参数:
    ----------
    references : np.ndarray
        形状为(源数量, 通道数, 采样点数)的3D数组,表示参考源信号
        
    estimates : np.ndarray  
        形状为(源数量, 通道数, 采样点数)的3D数组,表示预测的源信号
        
    返回:
    -------
    np.ndarray
        包含每个源的SDR值的1D数组
    """
    eps = 1e-8  # 避免数值错误的小量
    num = np.sum(np.square(references), axis=(1, 2))  # 计算参考信号能量
    den = np.sum(np.square(references - estimates), axis=(1, 2))  # 计算误差能量
    num += eps
    den += eps
    return 10 * np.log10(num / den)  # 转换为分贝单位


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    计算尺度不变信号失真比(Scale-Invariant Signal-to-Distortion Ratio, SI-SDR)
    
    SI-SDR是SDR的变体,对预测信号相对于参考信号的缩放不敏感。
    它通过将预测信号缩放到与参考信号匹配后再计算SDR。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(源数量, 通道数, 采样点数)的3D数组,表示参考源信号
        
    estimate : np.ndarray
        形状为(源数量, 通道数, 采样点数)的3D数组,表示预测的源信号
        
    返回:
    -------
    float
        以分贝(dB)为单位的SI-SDR标量值
    """
    eps = 1e-8  # 避免数值错误
    # 计算最优缩放因子
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference ** 2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))  # 重塑为[源数量, 1]

    # 应用缩放并计算SI-SDR
    reference = reference * scale
    si_sdr = np.mean(10 * np.log10(
        np.sum(reference ** 2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))

    return si_sdr


def L1Freq_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        fft_size: int = 2048,
        hop_size: int = 1024,
        device: str = 'cpu'
) -> float:
    """
    计算参考信号和预测信号之间的L1频率度量
    
    该指标通过短时傅里叶变换(STFT)比较参考和预测音频信号的幅度谱,
    计算它们之间的L1损失。结果被缩放到[0, 100]范围,值越高表示性能越好。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    fft_size : int, 可选
        STFT的窗口大小,默认2048
        
    hop_size : int, 可选
        STFT帧之间的跳跃大小,默认1024
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    float
        范围在[0, 100]的L1频率度量值
    """

    # 将numpy数组转换为torch张量并移动到指定设备
    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    # 计算STFT,获取复数形式的频谱
    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)

    # 计算幅度谱(取复数的模)
    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    # 计算L1损失并缩放
    # 乘以10是为了调整损失的尺度,使最终结果更有区分度
    loss = 10 * F.l1_loss(estimate_mag, reference_mag)

    # 将损失值转换到[0,100]范围,损失越小,返回值越接近100
    ret = 100 / (1. + float(loss.cpu().numpy()))

    return ret


def NegLogWMSE_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        mixture: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    计算参考信号、预测信号和混合信号之间的对数加权均方误差(Log-WMSE)
    
    该指标评估音频源分离中预测信号相对于参考信号的质量。使用对数尺度有助于评估
    具有较大幅度差异的信号。结果为负值,值越大表示分离效果越好。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    mixture : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示混合音频信号
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    float
        负的对数加权均方误差值
    """
    # 导入LogWMSE损失函数
    from torch_log_wmse import LogWMSE
    
    # 初始化LogWMSE计算器
    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / 44100,  # 音频长度(秒)
        sample_rate=44100,  # 采样率44100Hz
        return_as_loss=False,  # 作为评估指标返回而非损失
        bypass_filter=False,  # 启用频率滤波
    )

    # 扩展维度并转换为torch张量
    # 增加batch维度和额外通道维度以匹配模型输入要求
    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)

    # 计算LogWMSE并返回负值(因为越小越好转换为越大越好)
    res = log_wmse(mixture, reference, estimate)
    return -float(res.cpu().numpy())


def AuraSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    使用短时傅里叶变换(STFT)损失计算参考信号和预测信号之间的频谱差异
    
    该指标同时考虑对数和线性幅度的STFT损失,常用于评估音频分离任务的质量。
    结果被缩放到[0, 100]范围,值越高表示分离效果越好。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    float
        范围在[0, 100]的STFT度量值
    """

    # 导入STFT损失函数
    from auraloss.freq import STFTLoss

    # 初始化STFT损失计算器
    stft_loss = STFTLoss(
        w_log_mag=1.0,  # 对数幅度权重
        w_lin_mag=0.0,  # 线性幅度权重
        w_sc=1.0,       # 谱质心权重
        device=device,
    )

    # 转换为torch张量并添加batch维度
    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)

    # 计算损失并缩放到[0,100]范围
    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())


def AuraMRSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    使用多分辨率短时傅里叶变换(MRSTFT)损失计算参考信号和预测信号之间的频谱差异
    
    该指标使用多个分辨率的STFT分析,能更好地表示音频信号的低频和高频成分。
    结果被缩放到[0, 100]范围,值越高表示分离效果越好。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    float
        范围在[0, 100]的MRSTFT度量值
    """

    # 导入多分辨率STFT损失函数
    from auraloss.freq import MultiResolutionSTFTLoss

    # 初始化多分辨率STFT损失计算器
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],  # 使用三种不同的FFT窗口大小
        hop_sizes=[256, 512, 1024],    # 对应的hop大小
        win_lengths=[1024, 2048, 4096], # 窗口长度
        scale="mel",  # 使用mel频率尺度
        n_bins=128,   # mel频率bins数量
        sample_rate=44100,
        perceptual_weighting=True,  # 使用感知权重
        device=device
    )

    # 转换为torch张量并添加batch维度
    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    # 计算损失并缩放到[0,100]范围
    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())


def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
) -> Tuple[float, float]:
    """
    计算参考信号和预测信号之间的'泄漏'和'完整度'指标
    
    'bleedless'指标衡量预测信号对参考信号的泄漏程度,
    'fullness'指标衡量预测信号相对于参考信号的完整性,
    两者都使用mel频谱图和分贝尺度进行计算。
    
    参数:
    ----------
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    sr : int, 可选
        采样率,默认44100Hz
        
    n_fft : int, 可选
        STFT的FFT大小,默认4096
        
    hop_length : int, 可选
        STFT的hop长度,默认1024
        
    n_mels : int, 可选
        mel频率bins数量,默认512
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    tuple
        包含两个值:
        - bleedless (float): 泄漏指标得分(越高越好)
        - fullness (float): 完整度指标得分(越高越好)
    """

    # 导入分贝转换函数
    from torchaudio.transforms import AmplitudeToDB

    # 转换为torch张量
    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    # 创建Hann窗
    window = torch.hann_window(n_fft).to(device)

    # 使用Hann窗计算STFT
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    # 创建mel滤波器组
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    # 计算mel频谱图
    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    # 转换为分贝尺度
    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    # 计算频谱差异
    diff = S2_db - S1_db

    # 分别处理正差异(泄漏)和负差异(不完整)
    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    # 计算平均差异
    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    # 计算最终得分
    bleedless = 100 * 1 / (average_positive + 1)  # 泄漏得分
    fullness = 100 * 1 / (-average_negative + 1)  # 完整度得分

    return bleedless.cpu().numpy(), fullness.cpu().numpy()


def get_metrics(
        metrics: List[str],
        reference: np.ndarray,
        estimate: np.ndarray,
        mix: np.ndarray,
        device: str = 'cpu',
) -> Dict[str, float]:
    """
    计算音频源分离模型性能评估指标
    
    根据指定的指标列表,计算参考信号、预测信号和混合信号之间的各种评估指标。
    
    参数:
    ----------
    metrics : List[str]
        要计算的指标名称列表(如['sdr', 'si_sdr', 'l1_freq'])
        
    reference : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示参考音频信号
        
    estimate : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示预测音频信号
        
    mix : np.ndarray
        形状为(通道数, 采样点数)的2D数组,表示混合音频信号
        
    device : str, 可选
        计算设备('cpu'或'cuda'),默认'cpu'
        
    返回:
    -------
    Dict[str, float]
        包含所有计算指标的字典
    """
    result = dict()

    # 调整所有输入信号的长度使其相同
    min_length = min(reference.shape[1], estimate.shape[1])
    reference = reference[..., :min_length]
    estimate = estimate[..., :min_length]
    mix = mix[..., :min_length]

    # 根据指标列表计算各项指标
    if 'sdr' in metrics:
        # 信号失真比
        references = np.expand_dims(reference, axis=0)
        estimates = np.expand_dims(estimate, axis=0)
        result['sdr'] = sdr(references, estimates)[0]

    if 'si_sdr' in metrics:
        # 尺度不变信号失真比
        result['si_sdr'] = si_sdr(reference, estimate)

    if 'l1_freq' in metrics:
        # L1频率度量
        result['l1_freq'] = L1Freq_metric(reference, estimate, device=device)

    if 'neg_log_wmse' in metrics:
        # 负对数加权均方误差
        result['neg_log_wmse'] = NegLogWMSE_metric(reference, estimate, mix, device)

    if 'aura_stft' in metrics:
        # STFT损失指标
        result['aura_stft'] = AuraSTFT_metric(reference, estimate, device)

    if 'aura_mrstft' in metrics:
        # 多分辨率STFT损失指标
        result['aura_mrstft'] = AuraMRSTFT_metric(reference, estimate, device)

    if 'bleedless' in metrics or 'fullness' in metrics:
        # 计算泄漏和完整度指标
        bleedless, fullness = bleed_full(reference, estimate, device=device)
        if 'bleedless' in metrics:
            result['bleedless'] = bleedless
        if 'fullness' in metrics:
            result['fullness'] = fullness

    return result