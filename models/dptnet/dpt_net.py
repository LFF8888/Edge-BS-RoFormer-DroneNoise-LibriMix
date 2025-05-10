import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import DPT_base, Encoder, Decoder
from .transformer_improved import TransformerEncoderLayer
from .utils import overlap_and_add

class DPTNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_cfg = config['audio']
        
        # Extract parameters from YAML config
        self.enc_dim = audio_cfg['dim_f']        # Frequency dimension (e.g., 1024)
        self.win_len = audio_cfg['n_fft']        # Window length (e.g., 2048)
        self.hop_length = audio_cfg['hop_length']  # Hop length (e.g., 512)
        self.chunk_size = audio_cfg['chunk_size']  # Input length (e.g., 131584)
        
        # Fixed architectural parameters (not in YAML)
        self.feature_dim = 256
        self.hidden_dim = 512
        self.layer = 6
        self.segment_size = 250
        self.num_spk = 1  # Single source for denoising
        
        # Components
        self.encoder = Encoder(W=self.win_len, N=self.enc_dim)
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)
        self.separator = BF_module(
            self.enc_dim, self.feature_dim, self.hidden_dim,
            self.num_spk, self.layer, self.segment_size
        )
        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = Decoder(self.enc_dim, self.win_len)

    def forward(self, x):
        """
        Input:
            x: (B, C, T) - e.g., (batch, 1, 131584)
        Output:
            (B, num_spk, C, T) - e.g., (batch, 1, 1, 131584)
        """
        # Store original input length
        input_T = x.shape[-1]  # e.g., 131584
        
        # Squeeze channel dimension
        x = x.squeeze(1)  # (B, C, T) -> (B, T)
        
        # Encoder
        mixture_w = self.encoder(x)  # [B, enc_dim, L], e.g., [B, 1024, 257]
        
        # Layer normalization
        score_ = self.enc_LN(mixture_w)
        
        # Separator
        score_ = self.separator(score_)  # [B, num_spk, T', feature_dim], e.g., [B, 1, 131072, 64]
        
        # Mask generation
        score_ = score_.view(-1, self.feature_dim, score_.shape[2])  # [B*num_spk, feature_dim, T']
        score = self.mask_conv1x1(score_)  # [B*num_spk, enc_dim, L]
        score = score.view(-1, self.num_spk, self.enc_dim, score.shape[-1])  # [B, num_spk, enc_dim, L]
        est_mask = F.relu(score)  # [B, num_spk, enc_dim, L]
        
        # Decoder
        est_source = self.decoder(mixture_w, est_mask)  # [B, num_spk, T'], e.g., [B, 1, 131072]
        
        # Adjust output length to match input
        current_T = est_source.shape[-1]
        if current_T < input_T:
            pad_length = input_T - current_T
            est_source = F.pad(est_source, (0, pad_length))  # Pad with zeros
        elif current_T > input_T:
            est_source = est_source[:, :, :input_T]  # Crop to input length
        
        # Format output as (B, num_spk, C, T)
        output = est_source.unsqueeze(2)  # [B, num_spk, 1, T]
        return output

class BF_module(DPT_base):
    def __init__(self, enc_dim, feature_dim, hidden_dim, num_spk, layer, segment_size):
        super().__init__(
            input_dim=enc_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_spk=num_spk,
            layer=layer,
            segment_size=segment_size
        )
        self.output = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # Bottleneck
        enc_feature = self.BN(input)  # [B, enc_dim, L] -> [B, feature_dim, L]
        
        # Split into segments
        enc_segments, rest = self.split_feature(enc_feature, self.segment_size)
        
        # Dual-path transformer
        batch = enc_segments.shape[0]
        output = self.DPT(enc_segments)  # [B, feature_dim*num_spk, segment_size, K]
        output = self.merge_feature(output, rest)  # [B, feature_dim*num_spk, T']
        
        # Gated output
        bf_filter = self.output(output) * self.output_gate(output)
        return bf_filter.view(batch, self.num_spk, -1, self.feature_dim)

#### Unit Test
if __name__ == "__main__":
    config = {
        "audio": {
            "chunk_size": 131584,
            "dim_f": 1024,
            "dim_t": 515,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 1,
            "sample_rate": 16000,
            "min_mean_abs": 0.000
        }
    }
    
    # Test input
    B, C, T = 2, 1, config["audio"]["chunk_size"]
    x = torch.randn(B, C, T)
    
    # Model
    model = DPTNet(config)
    output = model(x)
    
    # Shape verification
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == (B, 1, C, T), f"Shape mismatch! Expected {(B, 1, C, T)}, got {output.shape}"
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
