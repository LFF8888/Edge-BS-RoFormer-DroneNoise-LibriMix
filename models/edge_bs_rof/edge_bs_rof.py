from functools import partial

import math  # Added for PositionalEncoding
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.edge_bs_rof.attend import Attend
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# 辅助函数

def exists(val):
    """
    检查输入值是否存在(不为None)
    Args:
        val: 任意输入值
    Returns:
        bool: 如果val不为None返回True,否则返回False
    """
    return val is not None


def default(v, d):
    """
    返回默认值,如果v存在则返回v,否则返回默认值d
    Args:
        v: 主要值
        d: 默认值
    Returns:
        如果v存在返回v,否则返回d
    """
    return v if exists(v) else d


def pack_one(t, pattern):
    """
    将单个张量按指定模式打包
    Args:
        t: 输入张量
        pattern: 打包模式字符串
    Returns:
        打包后的张量
    """
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    """
    将打包的张量解包并返回第一个元素
    Args:
        t: 打包的张量
        ps: 原始形状信息
        pattern: 解包模式字符串
    Returns:
        解包后的第一个张量
    """
    return unpack(t, ps, pattern)[0]


# 归一化层

def l2norm(t):
    """
    对输入张量进行L2归一化
    Args:
        t: 输入张量
    Returns:
        归一化后的张量
    """
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    """
    RMS(均方根)归一化层
    相比LayerNorm计算更简单,性能更好
    
    Args:
        dim: 归一化的维度大小
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class PositionalEncoding(Module):
    """
    绝对位置编码
    
    Args:
        dim: 编码维度
        max_seq_len: 最大序列长度
    """
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, dim]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


# 注意力机制相关模块

class FeedForward(Module):
    """
    前馈神经网络
    包含两个线性层和GELU激活函数,用于特征转换
    
    Args:
        dim: 输入维度
        mult: 隐藏层维度扩展倍数
        dropout: dropout比率
    """
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    """
    多头注意力机制
    支持旋转位置编码和Flash Attention优化
    
    Args:
        dim: 输入维度
        heads: 注意力头数
        dim_head: 每个注意力头的维度
        dropout: dropout比率
        rotary_embed: 旋转位置编码实例
        flash: 是否使用Flash Attention优化
        use_rotary_pos: 是否使用旋转位置编码
    """
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            use_rotary_pos=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed  # 旋转位置编码
        self.use_rotary_pos = use_rotary_pos  # 是否使用旋转位置编码

        self.attend = Attend(flash=flash, dropout=dropout)  # 注意力计算核心模块

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)  # 生成查询(Q)、键(K)、值(V)的线性层

        self.to_gates = nn.Linear(dim, heads)  # 生成注意力门控权重的线性层

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        # 生成Q,K,V并重排维度
        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        # 应用旋转位置编码
        if self.use_rotary_pos and exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        # 计算注意力
        out = self.attend(q, k, v)

        # 应用门控机制
        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        # 输出投影
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    """
    线性注意力机制
    将传统注意力的计算复杂度从O(n^2)降到O(n)
    基于论文 https://arxiv.org/abs/2106.09681
    
    Args:
        dim: 输入维度
        dim_head: 每个注意力头的维度
        heads: 注意力头数
        scale: 缩放因子
        flash: 是否使用Flash Attention优化
        dropout: dropout比率
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads  # 计算内部维度
        self.norm = RMSNorm(dim)      # 层归一化

        # 生成Q,K,V的线性变换和重排维度
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),  # 线性投影到Q,K,V空间
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)  # 重排维度以适应多头注意力
        )

        # 可学习的温度参数,用于缩放注意力分数
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        # 注意力计算模块
        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash
        )

        # 输出投影层
        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),  # 重排维度
            nn.Linear(dim_inner, dim, bias=False)  # 线性投影回原始维度
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)  # 输入归一化

        # 生成Q,K,V向量
        q, k, v = self.to_qkv(x)

        # 对Q,K进行L2归一化,增强稳定性
        q, k = map(l2norm, (q, k))
        # 应用温度缩放
        q = q * self.temperature.exp()

        # 计算注意力
        out = self.attend(q, k, v)

        # 输出投影
        return self.to_out(out)


class Transformer(Module):
    """
    Transformer编码器,由多层注意力层和前馈网络组成
    
    Args:
        dim: 输入维度
        depth: 层数
        dim_head: 注意力头维度
        heads: 注意力头数
        attn_dropout: 注意力dropout率
        ff_dropout: 前馈网络dropout率
        ff_mult: 前馈网络隐藏层维度倍数
        norm_output: 是否对输出进行归一化
        rotary_embed: 旋转位置编码
        flash_attn: 是否使用Flash Attention
        linear_attn: 是否使用线性注意力
        use_rotary_pos: 是否使用旋转位置编码（RoPE）
        max_seq_len: 最大序列长度
    """
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            use_rotary_pos=False,  # 是否使用旋转位置编码
            max_seq_len=1000       # 最大序列长度
    ):
        super().__init__()
        self.layers = ModuleList([])  # 存储所有层
        self.use_rotary_pos = use_rotary_pos
        self.max_seq_len = max_seq_len

        # 初始化绝对位置编码
        if not self.use_rotary_pos:
            self.positional_encoding = PositionalEncoding(dim=dim, max_seq_len=max_seq_len)
        else:
            self.positional_encoding = None

        # 构建多层结构
        for _ in range(depth):
            if linear_attn:
                # 使用线性注意力层
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                # 使用标准注意力层
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                 rotary_embed=rotary_embed, flash=flash_attn, use_rotary_pos=use_rotary_pos)
            # 每层包含注意力模块和前馈网络
            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        # 输出归一化层
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        if not self.use_rotary_pos:
            x = self.positional_encoding(x)
        # 依次通过每一层,使用残差连接
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力层 + 残差
            x = ff(x) + x    # 前馈网络 + 残差

        return self.norm(x)  # 输出归一化


# 频带分割模块：该模块用于将输入的特征张量按照预定义的频带维数进行拆分，
# 并对每个频带分别进行特征提取映射，最终堆叠各频带输出以便后续处理。
class BandSplit(Module):
    """
    将输入按频带分割并映射到指定维度

    参数：
        dim: 映射后每个频带输出特征的目标维度。
        dim_inputs: 一个元组，每个元素表示对应频带在输入数据中所占的维数。
                    例如，若dim_inputs=(a, b, c)，则表示输入张量最后一维依次拆分为a, b, c三个部分，
                    分别对应三个频带。
    """
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs  # 保存各频带对应的输入维数信息
        self.to_features = ModuleList([])  # 用于存储每个频带对应的特征提取网络

        # 针对每个频带构建单独的特征提取网络
        for dim_in in dim_inputs:
            # 构建一个顺序模型：
            # 1. 首先使用RMSNorm对输入数据进行归一化，稳定数值分布；
            # 2. 然后使用nn.Linear层将输入数据从原始维数dim_in映射到目标维度dim。
            net = nn.Sequential(
                RMSNorm(dim_in),   # 对当前频带输入进行归一化
                nn.Linear(dim_in, dim)  # 将输入进行线性变换映射到目标维度
            )
            # 将构建的网络添加到ModuleList中
            self.to_features.append(net)

    def forward(self, x):
        """
        前向传播：
        1. 根据预先设定的dim_inputs，将输入张量x在最后一个维度上进行切分，得到多个频带数据。
        2. 针对每个频带，利用其对应的特征提取网络进行处理，提取出目标特征。
        3. 将所有频带处理后的结果在新添加的倒数第二个维度上进行堆叠，
           以便后续模块能够识别各频带信息。
        """
        # 按最后一维依据dim_inputs的数值拆分输入张量，每个拆分对应一个频带
        x = x.split(self.dim_inputs, dim=-1)

        outs = []  # 用于存放各个频带处理后的输出
        # 遍历每个频带的输入和对应的特征提取网络
        for split_input, to_feature in zip(x, self.to_features):
            # 对单个频带数据进行特征提取
            split_output = to_feature(split_input)
            outs.append(split_output)

        # 将各个频带的输出在新的维度（倒数第二维）上堆叠，生成最终输出
        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    """
    多层感知机（MLP）模块：
    使用多个线性层和激活函数堆叠实现非线性映射，用于将输入特征转换到指定的输出维度。

    参数：
        dim_in: 输入特征的维度。
        dim_out: 输出特征的目标维度。
        dim_hidden: 隐藏层的维度，若未指定则默认等于dim_in。
        depth: 网络深度，即总共包含几个线性层（至少为1）。
        activation: 激活函数，默认采用nn.Tanh。
    """
    # 如果未指定隐藏层尺寸，则使用输入特征尺寸作为默认值
    dim_hidden = default(dim_hidden, dim_in)

    net = []  # 用于存储构建的各层
    # 生成层次尺寸序列：输入层，depth-1个隐藏层，和输出层
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    # 构建每一层的网络：依次添加线性层，且非最后一层后加激活函数
    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = (ind == (len(dims) - 2))  # 判断当前是否为最后一层

        # 添加线性映射层：将输入从layer_dim_in转换为layer_dim_out
        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        # 如果当前层不是最后一层，则添加激活函数以引入非线性
        if not is_last:
            net.append(activation())

    # 将所有构造的层按顺序包裹为一个Sequential模块，并返回
    return nn.Sequential(*net)


class MaskEstimator(Module):
    """
    掩码估计模块：
    该模块利用多层感知机和GLU门控机制对输入特征进行处理，
    估计出每个频带对应的音频分离掩码。
    """
    @beartype
    def __init__(self, dim, dim_inputs: Tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs  # 保存各频带输入特征维数信息
        self.to_freqs = ModuleList([])  # 用于存储每个频带的频率映射网络
        # 计算隐藏层维数：通常为基本维度dim乘以一个扩展因子
        dim_hidden = dim * mlp_expansion_factor

        # 为每个频带构建映射网络，每个网络由MLP模块和GLU激活构成
        for dim_in in dim_inputs:
            # 构造Sequential网络：
            # a. 使用MLP将输入从dim映射到(dim_in * 2)维度，
            # b. 通过nn.GLU在最后一个维度上进行门控操作，增强映射复杂性
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )
            self.to_freqs.append(mlp)

    def forward(self, x):
        """
        前向传播：
        1. 将输入张量x按照倒数第二个维度拆分成各个频带的数据。
        2. 对每个频带分别通过对应的映射网络生成分离掩码信息。
        3. 将所有频带的输出在最后一个维度上拼接成联合输出。
        """
        # 使用unbinding拆分x，使得每个拆分元素代表一个频带的特征
        x = x.unbind(dim=-2)

        outs = []  # 用于存放每个频带经过网络映射后的结果
        # 遍历每个频带的数据和对应的网络
        for band_features, mlp in zip(x, self.to_freqs):
            # 利用当前网络估计当前频带的频率映射输出
            freq_out = mlp(band_features)
            outs.append(freq_out)

        # 沿最后一个维度将所有频带的输出拼接为一个整体
        return torch.cat(outs, dim=-1)


# 主模型

# 默认的频带划分配置：
# 定义每个频带中包含的频率数量，提供多尺度频率分辨率信息，
# 例如前24个频带中每个频带包含2个频率，后续频带数目依次增多。
DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(Module):
    """
    基于Roformer的音源分离模型
    使用多层Transformer对时频表示进行建模
    支持单声道/立体声输入
    可以分离多个音源
    """

    @beartype
    def __init__(
            self,
            dim,                        # 模型的基本特征维度
            *,
            depth,                      # Transformer层的数量
            stereo=False,               # 是否处理立体声音频（True代表立体声，即2通道；False代表单声道，即1通道）
            num_stems=1,                # 要分离的音源数量（例如单人或多人说话）
            time_transformer_depth=2,   # 处理时间信息的Transformer模块深度
            freq_transformer_depth=2,   # 处理频率信息的Transformer模块深度
            linear_transformer_depth=0, # 线性注意力模块的深度（若为0则不使用）
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # 每个频带包含的频率数配置
            dim_head=64,                # 多头注意力中每个头的维度
            heads=8,                    # Transformer注意力头的数量
            attn_dropout=0.,            # 注意力层中的dropout概率
            ff_dropout=0.,              # 前馈层（MLP）中的dropout概率
            flash_attn=True,            # 是否使用flash attention来加速注意力计算
            dim_freqs_in=1025,          # STFT后得到的频率数（通常由STFT参数决定）
            stft_n_fft=2048,            # STFT的FFT窗口大小
            stft_hop_length=512,        # STFT的跳步长度（即每帧之间的间隔）
            stft_win_length=2048,       # STFT窗口的长度
            stft_normalized=False,      # 是否对STFT结果进行归一化
            stft_window_fn: Optional[Callable] = None,  # 生成STFT窗函数的方法
            mask_estimator_depth=2,     # 掩码估计器中Transformer的深度
            multi_stft_resolution_loss_weight=1.,  # 多分辨率STFT损失的权重
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),  # 多分辨率下不同窗口尺寸的配置
            multi_stft_hop_size=147,    # 多分辨率STFT使用的跳步长度
            multi_stft_normalized=False,  # 多分辨率STFT是否归一化
            multi_stft_window_fn: Callable = torch.hann_window,  # 多分辨率STFT窗函数，默认使用汉宁窗
            mlp_expansion_factor=4,     # MLP扩展因子，用于控制隐藏层的宽度
            use_torch_checkpoint=False, # 是否使用torch的checkpoint机制减少中间内存占用
            skip_connection=False,      # 是否在Transformer模块间使用跳跃连接（残差连接）
            use_rotary_pos=False,       # 是否使用旋转位置编码（RoPE）
            max_seq_len=1000            # 最大序列长度
    ):
        super().__init__()  # 初始化父类Module

        # 根据stereo参数确定音频通道数，立体声为2通道，单声道为1通道
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems  # 保存要分离的音源数量
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.use_rotary_pos = use_rotary_pos
        self.max_seq_len = max_seq_len

        # 初始化一个ModuleList用于存储多层Transformer模块
        self.layers = ModuleList([])

        # 定义Transformer的共有参数，这些参数会传递给每一层的Transformer模块
        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,  # 输出不做归一化，由final_norm负责归一化
            use_rotary_pos=use_rotary_pos,
            max_seq_len=max_seq_len
        )

        # 初始化时间轴和频率轴上的旋转位置编码器，为Transformer提供位置信息
        time_rotary_embed = RotaryEmbedding(dim=dim_head) if use_rotary_pos else None
        freq_rotary_embed = RotaryEmbedding(dim=dim_head) if use_rotary_pos else None

        # 根据深度构造多个Transformer层，每个层可以包含一组不同注意力的模块
        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                # 如果设定了线性Transformer层，则添加线性注意力模块
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            # 添加面向时间维度的Transformer，用于捕捉时间相关性
            tran_modules.append(
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            # 添加面向频率维度的Transformer，用于捕捉频率相关性
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            )
            # 将当前层的模块列表封装为ModuleList后添加到整体的层中
            self.layers.append(nn.ModuleList(tran_modules))

        # 对所有Transformer层的输出进行归一化处理
        self.final_norm = RMSNorm(dim)

        # 配置STFT转换的参数，将时域音频转换到频域
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        # 设置STFT窗函数生成方法，若未提供则使用默认的torch.hann_window，并设定固定窗口长度
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # 通过对一段随机信号做STFT来计算转换后的频率数
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        # 确保传入的频带配置至少有两个频带，并且所有频带的频率总数等于STFT输出的频率数
        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs, f'频带数量必须等于STFT设置下的频率数{freqs},但得到{sum(freqs_per_bands)}'

        # 针对复数数据表示和音频通道数进行处理，将每个频带的频率数乘以2（表示实部和虚部）及音频通道数
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        # 初始化频带分割模块，将输入的频谱数据划分为多个预设的频带
        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        # 为每个音源构建一个掩码估计器，后续用于估计每个音源的掩码
        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        # 设置多分辨率STFT损失相关参数，以多分辨率方式衡量重构音频与目标音频的差异
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        # 配置多分辨率STFT的其他参数，如跳步长度和归一化选项
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,  # 输入的原始时域音频，其形状可以为[b, t]或[b, s, t]
            target=None,  # 目标音频，用于在训练时计算损失
            return_loss_breakdown=False  # 是否返回损失的各个组成部分
    ):
        """
        前向传播过程
        
        输入维度说明:
        b - batch size (批次大小)
        f - 频率数
        t - 时间帧数
        s - 音频通道数 (单声道1个通道，立体声2个通道)
        n - 音源数量 (分离目标的数量)
        c - 复数表示的维度 (2，代表实部和虚部)
        d - Transformer内部的特征维度
        """
        device = raw_audio.device  # 获取输入音频所在的设备（例如CPU、GPU等）

        # 检查是否在MacOS的MPS设备上运行，以应对FFT操作可能存在的兼容性问题
        x_is_mps = True if device.type == "mps" else False

        # 如果输入音频维度为2，则表示缺少通道维度，故将其扩展（例如[b, t]扩展为[b, 1, t]）
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]  # 获取输入音频的通道数
        # 检查输入通道数是否和模型配置匹配：单声道应为1通道，立体声应为2通道
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), '立体声设置必须与输入音频通道数匹配'

        # 对输入的时域音频进行STFT变换，将其转换到频域以便后续处理
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
        stft_window = self.stft_window_fn(device=device)  # 根据设备创建STFT窗函数

        # 执行STFT操作，为了处理可能在MacOS MPS平台上的兼容性问题，采用try/except结构
        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs,
                                   window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(device)
        # 将复数STFT输出转换为实数张量（最后一维为实部和虚部）
        stft_repr = torch.view_as_real(stft_repr)

        # 恢复STFT输出的原始打包形状
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # 将不同音频通道（如立体声的两个通道）与频率轴合并，为后续处理合并信息
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # 调整张量形状，使得时间轴成为第一维，方便Transformer模块处理
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        # 通过频带分割模块，将频谱数据按照预设配置划分为多个频带
        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # 进入多层Transformer模块进行轴向（时间/频率）注意力处理
        store = [None] * len(self.layers)  # 用于存储每层的输出（如启用跳跃连接时会用到）
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                # 如果当前层包含三个子模块，表示包含线性注意力模块
                linear_transformer, time_transformer, freq_transformer = transformer_block

                # 对x进行打包，携带形状信息，便于后续恢复原始形状
                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    # 使用checkpoint机制以节省内存
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                # 解包恢复原始形状
                x, = unpack(x, ft_ps, 'b * d')
            else:
                # 否则说明当前层只包含时间和频率两个维度的Transformer
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                # 如果启用了跳跃连接，则将之前各层的输出累加到当前输出上
                for j in range(i):
                    x = x + store[j]

            # 对时间维度进行Transformer处理：先交换维度使时间维度位于合适的位置
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)
            x, = unpack(x, ps, '* t d')

            # 对频率维度进行Transformer处理：重新排列维度，使频率维度为目标
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                # 保存当前层的输出以供后续的跳跃连接使用
                store[i] = x

        # 对所有Transformer层的输出进行最终归一化操作
        x = self.final_norm(x)

        # 记录实际使用的音源数量，即掩码估计器的个数
        num_stems = len(self.mask_estimators)

        # 对每个音源进行分离掩码的估计，使用checkpoint机制（如果启用）可以降低内存占用
        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        # 调整掩码张量的形状，将最后一维分解为频率和复数（实部、虚部）两个部分
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # 为后续的调制操作，将原始STFT表示增加一个音源维度
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # 将原始STFT和掩码的实数形式转换为复数形式，便于频域操作
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        # 利用估计的掩码调制原始频谱，完成频域的音源分离
        stft_repr = stft_repr * mask

        # 调整分离后的频谱形状，为逆STFT做准备，恢复每个音源和音频通道的独立维度
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        # 执行逆STFT操作，将频域数据恢复成时域音频，同时考虑MacOS MPS的兼容性
        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)
        # 调整逆变换后音频的形状，将多音源和通道分离开来
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        # 如果只有一个音源，则移除音源维度
        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # 若未提供目标音频，则直接返回重构后的音频
        if not exists(target):
            return recon_audio

        # 当模型设计为分离多个音源时，检查目标音频的维度是否符合要求（4维且第一维大小与音源数一致）
        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        # 如果目标音频仅为二维，则扩展为三维（增加音源维度）
        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        # 截断目标音频的时间长度，以确保和逆STFT生成的音频长度一致
        target = target[..., :recon_audio.shape[-1]]

        # 计算基础的L1损失，用于衡量重构音频和目标音频之间的绝对差异
        loss = F.l1_loss(recon_audio, target)

        # 初始化多分辨率STFT损失，用于捕捉在不同频率分辨率下存在的细微误差
        multi_stft_resolution_loss = 0.
        for window_size in self.multi_stft_resolutions_window_sizes:
            # 为当前窗口尺寸配置STFT参数，确保FFT长度不小于窗口尺寸
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )
            # 对重构音频与目标音频分别进行STFT变换
            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)
            # 累加各分辨率下的L1损失
            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        # 将多分辨率STFT损失乘以预设权重，并与基础L1损失相加得到总损失
        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        total_loss = loss + weighted_multi_resolution_loss

        # 若不需要返回详细的损失分解，则直接返回总损失
        if not return_loss_breakdown:
            return total_loss

        # 返回总损失以及（L1损失, 多分辨率STFT损失）的详细信息
        return total_loss, (loss, multi_stft_resolution_loss)