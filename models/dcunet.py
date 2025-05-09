import torch
import torch.nn as nn
import torch.nn.functional as F

class CConv2d(nn.Module):
    """复数卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.im_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        """
        输入x: (batch, channels, freq, time, 2)
        输出: (batch, channels, freq, time, 2)
        """
        # 分离实部和虚部
        x_real, x_im = x[..., 0], x[..., 1]
        
        # 应用卷积
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        # 合并实部和虚部
        return torch.stack([c_real, c_im], dim=-1)

class CConvTranspose2d(nn.Module):
    """复数转置卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        self.real_convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                           stride, padding, output_padding)
        self.im_convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                         stride, padding, output_padding)
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        """
        输入x: (batch, channels, freq, time, 2)
        输出: (batch, channels, freq, time, 2)
        """
        # 分离实部和虚部
        x_real, x_im = x[..., 0], x[..., 1]
        
        # 应用转置卷积
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        # 合并实部和虚部
        return torch.stack([ct_real, ct_im], dim=-1)

class CBatchNorm2d(nn.Module):
    """复数批归一化"""
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.im_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        输入x: (batch, channels, freq, time, 2)
        输出: (batch, channels, freq, time, 2)
        """
        # 分离实部和虚部
        x_real, x_im = x[..., 0], x[..., 1]
        
        # 应用批归一化
        x_real = self.real_bn(x_real)
        x_im = self.im_bn(x_im)
        
        # 合并实部和虚部
        return torch.stack([x_real, x_im], dim=-1)

class Encoder(nn.Module):
    """编码器模块"""
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.cconv = CConv2d(in_channels, out_channels, kernel, stride, padding)
        self.cbn = CBatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.cconv(x)
        x = self.cbn(x)
        return self.act(x)

class Decoder(nn.Module):
    """解码器模块"""
    def __init__(self, in_channels, out_channels, kernel, stride, output_padding, padding, last_layer=False):
        super().__init__()
        self.cconvt = CConvTranspose2d(in_channels, out_channels, kernel, stride, output_padding, padding)
        self.cbn = CBatchNorm2d(out_channels) if not last_layer else None
        self.act = nn.LeakyReLU() if not last_layer else None
        self.last_layer = last_layer

    def forward(self, x):
        x = self.cconvt(x)
        if not self.last_layer:
            x = self.cbn(x)
            x = self.act(x)
        else:
            m_phase = x / (torch.abs(x) + 1e-8)
            m_mag = torch.tanh(torch.abs(x))
            x = m_phase * m_mag
        return x

class STFTProcessor(nn.Module):
    """STFT处理模块 与其他模型保持统一接口"""
    def __init__(self, config):
        super().__init__()
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.window = torch.hann_window(self.n_fft)
        self.dim_f = config['audio']['dim_f']

    def transform(self, x):
        """
        输入x: (batch, channels, time)
        输出: (batch, 1, freq, time, 2)
        """
        if __name__ == "__main__":
            print(f"STFT输入形状: {x.shape}")
        x = x.squeeze(1)  # 移除通道维度
        if __name__ == "__main__":
            print(f"移除通道维度后: {x.shape}")
        
        # 执行STFT
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                      window=self.window.to(x.device), return_complex=True,
                      normalized=True)
        if __name__ == "__main__":
            print(f"STFT后: {X.shape}")
        
        # 转换为实数表示并调整维度
        X = torch.view_as_real(X)  # (batch, freq, time, 2)
        if __name__ == "__main__":
            print(f"转为实数表示后: {X.shape}")
        X = X.unsqueeze(1)  # 添加通道维度 (batch, 1, freq, time, 2)
        if __name__ == "__main__":
            print(f"添加通道维度后: {X.shape}")
        
        return X

    def inverse(self, X):
        """
        输入X: (batch, 1, freq, time, 2)
        输出: (batch, channels=1, time)
        """
        if __name__ == "__main__":
            print(f"ISTFT输入形状: {X.shape}")
        # 调整维度以适配ISTFT
        X = X.squeeze(1)  # 移除通道维度 (batch, freq, time, 2)
        if __name__ == "__main__":
            print(f"移除通道维度后: {X.shape}")
        X = torch.view_as_complex(X)
        if __name__ == "__main__":
            print(f"转为复数后: {X.shape}")
        
        x = torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=self.window.to(X.device), normalized=True)
        if __name__ == "__main__":
            print(f"ISTFT后: {x.shape}")
        
        x = x.unsqueeze(1)  # 添加通道维度 (batch, 1, time)
        if __name__ == "__main__":
            print(f"添加通道维度后: {x.shape}")
        return x

class DCUNet(nn.Module):
    """深度复数U-Net主模型"""
    def __init__(self, config):
        super().__init__()
        # 添加STFT处理器
        self.stft = STFTProcessor(config)
        
        # 调整输入输出通道
        self.input_channels = 1  # 修改为1，因为复数通道在最后一维
        self.output_channels = config['audio']['num_channels']
        
        # 固定参数（根据原始论文实现）
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        
        # 编码器 - 修改第一个Encoder的输入通道为1
        self.encoders = nn.ModuleList([
            Encoder(1, 45, (7,5), (2,2), padding=(3,2)),  # 修改输入通道为1并添加padding
            Encoder(45, 90, (7,5), (2,2), padding=(3,2)),
            Encoder(90, 90, (5,3), (2,2), padding=(2,1)),
            Encoder(90, 90, (5,3), (2,2), padding=(2,1)),
            Encoder(90, 90, (5,3), (2,1), padding=(2,1))
        ])
        
        # 解码器
        self.decoders = nn.ModuleList([
            Decoder(90, 90, (5,3), (2,1), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 90, (5,3), (2,2), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 90, (5,3), (2,2), output_padding=(0,0), padding=(2,1)),
            Decoder(180, 45, (7,5), (2,2), output_padding=(0,0), padding=(3,2)),
            Decoder(90, 1, (7,5), (2,2), output_padding=(0,1), padding=(3,2), last_layer=True)
        ])

    def forward(self, x):
        """
        输入x: (batch, channels, time)
        输出: (batch, instruments=1, channels=1, time)
        """
        if __name__ == "__main__":
            print("\n----- 开始前向传播 -----")
            print(f"输入形状: {x.shape}")
        
        # STFT变换
        X = self.stft.transform(x)  # (batch, 1, freq, time, 2)
        if __name__ == "__main__":
            print(f"\n----- 编码器过程 -----")
            print(f"STFT后形状: {X.shape}")
        
        # 编码过程
        encoder_features = []
        current = X
        
        for i, encoder in enumerate(self.encoders):
            current = encoder(current)
            if i < len(self.encoders) - 1:
                encoder_features.append(current)
            if __name__ == "__main__":
                print(f"编码器{i+1}输出: {current.shape}")
        
        if __name__ == "__main__":
            print(f"\n----- 解码器过程 -----")
        # 解码过程
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                current = decoder(current)
            else:
                skip_connection = encoder_features[-(i)]
                current = decoder(torch.cat([current, skip_connection], dim=1))
            if __name__ == "__main__":
                print(f"解码器{i+1}输出: {current.shape}")
        
        # 调整维度以匹配X进行掩码操作
        output = current * X
        if __name__ == "__main__":
            print(f"\n掩码后形状: {output.shape}")
        
        # ISTFT变换
        output = self.stft.inverse(output)  # (batch, 1, time)
        # 调整维度 (batch, instruments=1, channels=1, time)
        output = output.unsqueeze(1)
        if __name__ == "__main__":
            print(f"最终输出形状: {output.shape}")
            print("----- 前向传播结束 -----\n")
        
        return output

# ---------------------- 简单测试用例 ----------------------
if __name__ == "__main__":
    # 模拟配置
    config = {
        "audio": {
            "chunk_size": 131584,  # 仅用于生成测试输入
            "dim_f": 1024,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 1,     # 这里请设为1，保证与单声道 STFT 逻辑匹配
            "sample_rate": 16000,
        },
        "training": {
            "batch_size": 10
        }
    }

    # 初始化模型
    model = DCUNet(config)
    print("模型结构:")
    print(model)

    # 创建测试输入: (batch_size, 1, time)
    batch_size = config["training"]["batch_size"]
    channels   = config["audio"]["num_channels"]  # 通常设为1
    time_len   = config["audio"]["chunk_size"]
    x = torch.randn(batch_size, channels, time_len)

    # 前向传播
    print("\n进行前向传播...")
    output = model(x)

    # 输出形状
    print("\n输出形状:", output.shape)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总量: {total_params}")
