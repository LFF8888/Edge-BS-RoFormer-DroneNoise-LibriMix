from functools import wraps  # 导入wraps装饰器，用于保留被装饰函数的原始元数据（如函数名、文档字符串等）
from packaging import version  # 从packaging库中导入version模块，用于版本号的比较操作
from collections import namedtuple  # 从collections模块中导入namedtuple，用于创建具名元组，以便后续存储配置数据

import os  # 导入操作系统接口模块，便于后续获取系统信息（例如判断操作系统类型）
import torch  # 导入PyTorch库，用于张量计算和深度学习操作
from torch import nn, einsum  # 从torch中导入神经网络模块nn和Einstein求和函数einsum，用于简洁地表达张量运算
import torch.nn.functional as F  # 导入torch.nn.functional模块，提供各种函数式神经网络操作（如softmax、卷积等）

from einops import rearrange, reduce  # 从einops库中导入rearrange和reduce函数，用于对张量进行灵活的形状重排和降维操作

# 常量定义区域

# 定义FlashAttentionConfig具名元组，该元组包含三个布尔型的配置参数：
# enable_flash: 是否启用Flash Attention（加速版注意力计算）
# enable_math: 是否使用数学计算方式实现注意力
# enable_mem_efficient: 是否启用内存优化策略
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# 辅助函数定义

def exists(val):
    # 检查传入的变量是否存在（即不为None）
    return val is not None

def default(v, d):
    # 如果变量v存在，则返回v；否则返回默认值d
    return v if exists(v) else d

def once(fn):
    # 装饰器：确保被装饰的函数fn只会被调用一次，避免重复执行（例如避免多次打印提示信息）
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            # 若函数已调用过，则直接返回，不再执行
            return
        called = True  # 标记该函数已被调用
        return fn(x)   # 调用原始函数fn
    return inner

# 创建一个仅输出一次信息的print函数，用于防止重复打印相同提示
print_once = once(print)

# 注意力机制核心类实现
class Attend(nn.Module):
    """
    Attend类实现了注意力机制的核心功能，支持两种计算方式：
      1. 标准点积注意力（dot-product attention）
      2. Flash Attention（优化版注意力计算），这种方式可以大幅降低内存消耗并加速计算，
         不过需要PyTorch 2.0或更高版本的支持。
         
    参数说明：
      dropout - 在注意力计算中使用的dropout比率，用于随机丢弃部分注意力权重防止过拟合
      flash   - 布尔标志，指示是否启用Flash Attention优化模式
      scale   - 注意力分数的缩放因子，若未指定，默认使用1/sqrt(d)（其中d为特征维度）
    """
    def __init__(
        self,
        dropout = 0.,      # 注意力机制中的Dropout丢弃比率
        flash = False,     # 是否启用Flash Attention优化计算模式
        scale = None       # 注意力分数的缩放因子，若为None，则在计算中使用默认的1/sqrt(d)
    ):
        super().__init__()  # 初始化父类nn.Module
        self.scale = scale  # 存储缩放因子
        self.dropout = dropout  # 存储Dropout比率
        self.attn_dropout = nn.Dropout(dropout)  # 基于给定的dropout比率创建Dropout层，用于后续对注意力权重的随机丢弃

        self.flash = flash  # 存储是否启用Flash Attention的标志
        # 若启用Flash Attention，则要求当前PyTorch版本至少为2.0，否则会触发断言错误
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # 设置默认的CPU注意力配置，所有配置参数均为True
        self.cpu_config = FlashAttentionConfig(True, True, True)
        # 初始化CUDA配置参数为None，后续将根据CUDA设备属性进行设置
        self.cuda_config = None

        # 如果没有可用的CUDA设备或没有启用Flash Attention，则无需进一步设置CUDA相关配置，直接返回
        if not torch.cuda.is_available() or not flash:
            return

        # 获取当前CUDA设备的属性，用于判断GPU的计算能力
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        # 构造GPU设备的版本号，包括major和minor，用于比较计算能力（例如8.0）
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')

        if device_version >= version.parse('8.0'):
            if os.name == 'nt':
                # 如果在Windows操作系统上运行，即使GPU计算能力>=8.0，也选择使用数学计算或内存优化的注意力（不启用Flash Attention）
                print_once('Windows OS detected, using math or mem efficient attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                # 非Windows系统且GPU计算能力>=8.0的情况下，启用Flash Attention以充分利用硬件加速
                print_once('GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            # 若GPU计算能力低于8.0，则使用数学计算或内存优化的注意力方式，避免Flash Attention可能不兼容的问题
            print_once('GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        """实现Flash Attention的优化计算方法"""
        # 解包查询向量q的形状，并获取注意力头的数量、查询序列长度等信息，
        # 同时从键向量k获取key序列长度，并检测张量是否处于CUDA上以及所属设备
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # 如果用户自定义了缩放因子，则需要对查询向量q进行缩放调整
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5  # 默认的缩放因子，通常为1/sqrt(特征维度)
            q = q * (self.scale / default_scale)  # 按自定义比例调整q

        # 根据当前张量是否在CUDA上，选择对应的注意力配置（CUDA配置或CPU配置）
        config = self.cuda_config if is_cuda else self.cpu_config

        # 利用PyTorch 2.0中的scaled_dot_product_attention在指定的配置下执行Flash Attention计算
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.  # 在训练阶段应用dropout，推理时则关闭
            )

        return out

    def forward(self, q, k, v):
        """
        前向传播函数，计算注意力输出。
        使用Einstein求和约定，其中各维度的含义如下：
          b - 批次大小 (batch size)
          h - 注意力头数 (heads)
          n, i, j - 序列相关维度（例如查询和键的序列长度）
          d - 特征维度 (feature dimension)
        参数：
          q - 查询向量（query）
          k - 键向量（key）
          v - 值向量（value）
        """
        # 获取查询和键的序列长度，并确定查询所在的设备
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        # 如果未提供自定义的缩放因子，则使用默认值1/sqrt(特征维度)
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # 当启用Flash Attention模式时，调用flash_attn函数获得注意力输出
        if self.flash:
            return self.flash_attn(q, k, v)

        # 计算标准点积注意力的相似度分数：
        # 利用爱因斯坦求和约定对查询q和键k的向量进行内积运算，并乘以缩放因子
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # 对相似度分数应用softmax归一化，得到注意力权重
        attn = sim.softmax(dim=-1)
        # 使用Dropout层对注意力权重进行随机丢弃，以防止过拟合
        attn = self.attn_dropout(attn)

        # 使用爱因斯坦求和约定，根据归一化的注意力权重对值向量v进行加权求和，生成输出
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
