# Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement

[![Read in Chinese](https://img.shields.io/badge/中文版-README-blue.svg)](README_CN.md)

This repository contains the official implementation and the DroneNoise-LibriMix (DN-LM) dataset for the paper "Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement".

## Abstract

Addressing the significant challenge of speech enhancement in ultra-low Signal-to-Noise Ratio (SNR) scenarios in Unmanned Aerial Vehicle (UAV) voice communication, this study proposes an edge-deployed Band-Split Rotary Position Encoding Transformer (Edge-BS-RoFormer). Existing deep learning methods show significant limitations in suppressing dynamic UAV noise under edge computing constraints. These limitations mainly include insufficient modeling of harmonic features and high computational complexity. The proposed method employs a band-split strategy to partition the speech spectrum into non-uniform sub-bands, integrates a dual-dimension Rotary Position Encoding (RoPE) mechanism for joint time-frequency modeling, and adopts FlashAttention to optimize computational efficiency. Experiments on a self-constructed DroneNoise-LibriMix (DN-LM) dataset demonstrate that the proposed method achieves Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) improvements of 2.2 dB and 2.2 dB, and Perceptual Evaluation of Speech Quality (PESQ) enhancements of 0.15 and 0.11, respectively, compared to Deep Complex U-Net (DCUNet) and HTDemucs under -15 dB SNR conditions. Edge deployment tests reveal the model's memory footprint is under 500MB with a Real-Time Factor (RTF) of 0.33, fulfilling real-time processing requirements. This study provides a lightweight solution for speech enhancement in complex acoustic environments. Furthermore, the open-source dataset facilitates the establishment of standardized evaluation frameworks in the field.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2025edgebsroformer,
  title={Edge-BS-RoFormer: Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement},
  author={Liu, Feifan and Li, Muying and Guo, Luming and Guo, Hao and Cao, Jie and Zhao, Wei and Wang, Jun},
  journal={Drones}, % Or the actual journal/conference
  year={2025}, % Or the actual year
  % publisher={MDPI} % Or the actual publisher
}
```
 