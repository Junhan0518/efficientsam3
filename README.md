# EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3



## 1. Introduction

### Motivation

[SAM3](https://ai.meta.com/sam3) (Segment Anything Model 3) has introduced powerful **Promptable Concept Segmentation (PCS)** capabilities, enabling semantic understanding and temporal object tracking beyond traditional mask generation. However, SAM3's massive vision backbone and dense memory bank make it impractical for real-time, on-device applications where computational resources and latency constraints are critical.

**EfficientSAM3** addresses this challenge by distilling SAM3's capabilities into lightweight architectures suitable for edge devices, enabling high-quality concept segmentation on mobile phones, embedded systems, and resource-constrained platforms.

<p align="center">
  <img src="images/efficientsam3.svg" alt="EfficientSAM3 Architecture" width="100%">
</p>

### Dual Strategy

EfficientSAM3 employs a **two-pronged optimization approach**:

1. **Prompt-in-the-Loop Knowledge Distillation**: Transfer SAM3's encoder knowledge to efficient lightweight backbones ([RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), [EfficientViT](https://github.com/mit-han-lab/efficientvit)) while maintaining semantic understanding through prompt-guided distillation.

2. **Perceiver-based Memory Compression**: Replace SAM3's dense memory bank with a compact Perceiver-based module (inspired by [EdgeTAM](https://github.com/facebookresearch/EdgeTAM)) for efficient temporal tracking and video segmentation with minimal memory overhead.

---

## 2. Supported Models and Architecture

| Component | Model/Backbone | Purpose |
|-----------|----------------|---------|
| **Teacher Models** | [SAM](https://github.com/facebookresearch/segment-anything) (Segment Anything Model) | Foundation for image-level encoder distillation |
| | [SAM2](https://github.com/facebookresearch/sam2) | Temporal memory and video tracking distillation |
| | SAM3 | Promptable Concept Segmentation (PCS) capabilities |
| **Student Backbones** | [RepViT](https://github.com/THU-MIG/RepViT) (M0.9, M1.1, M2.3) | Mobile-optimized Vision Transformer for highest throughput |
| | [TinyViT](https://github.com/wkcn/TinyViT) (5M, 11M, 21M) | Balanced efficiency and performance |
| | [EfficientViT](https://github.com/mit-han-lab/efficientvit) (B0, B1, B2) | Ultra-lightweight architectures for minimal latency |
| **Reference Modules** | [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) | Lightweight encoder distillation techniques |
| | [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Perceiver-based memory compression for video tracking |

---

## 3. The Three-Stage Training Pipeline

EfficientSAM3 is trained through a carefully designed three-stage curriculum:

1. **Stage 1: Encoder Distillation (Image-Level PCS)**
   - Distill the SAM3 encoder to nine student backbones (3 RepViT × 3 TinyViT × 3 EfficientViT variants)
   - Use SAM1 image dataset with **Prompt-in-the-Loop Distillation**
   - Align student backbone features with teacher encoder outputs using prompt-guided attention
   - Output: Nine independently trained lightweight encoders

2. **Stage 2: Temporal Memory Distillation (Video Tracking)**
   - Replace SAM3's dense memory bank with a compact **Perceiver-based memory module** (adapted from [EdgeTAM](https://github.com/facebookresearch/EdgeTAM))
   - Distill memory-conditioned mask predictions using [SAM2](https://github.com/facebookresearch/sam2) video dataset
   - Train the Perceiver module to compress and retrieve spatiotemporal features efficiently
   - Align temporal tracking capabilities between teacher and student architectures

3. **Stage 3: End-to-End Fine-Tuning**
   - Refine the complete EfficientSAM3 pipeline using SAM3 official dataset
   - Joint optimization of distilled encoder + compressed memory + mask decoder
   - Preserve Promptable Concept Segmentation capabilities while maintaining efficiency
   - Output: Production-ready EfficientSAM3 models for on-device deployment

---

## 4. EfficientSAM3 Model Zoo & Weight Release

| Model Name | Backbone | Parameters | Stage 1 Weights<br/>(Encoder Distilled) | Stage 3 Weights<br/>(End-to-End Fine-Tuned) | Notes |
|------------|----------|------------|----------------------------------------|---------------------------------------------|-------|
| **EfficientSAM3-RepViT-S** | RepViT-M0.9 | 5.1M | $$\text{Planned}$$ | $$\text{Planned}$$ | Optimized for highest on-device throughput. |
| **EfficientSAM3-RepViT-M** | RepViT-M1.1 | 6.8M | $$\text{Planned}$$ | $$\text{Planned}$$ | |
| **EfficientSAM3-RepViT-L** | RepViT-M2.3 | 8.2M | $$\text{Planned}$$ | $$\text{Planned}$$ | |
| **EfficientSAM3-TinyViT-S** | TinyViT-5M | 5.4M | $$\text{Planned}$$ | $$\text{Planned}$$ | Balanced performance and parameter count. |
| **EfficientSAM3-TinyViT-M** | TinyViT-11M | 11M | $$\text{Planned}$$ | $$\text{Planned}$$ | |
| **EfficientSAM3-TinyViT-L** | TinyViT-21M | 21M | $$\text{Planned}$$ | $$\text{Planned}$$ | |
| **EfficientSAM3-EffViT-S** | EfficientViT-B0 | 0.7M | $$\text{Planned}$$ | $$\text{Planned}$$ | Ultra-lightweight variant for minimal latency. |
| **EfficientSAM3-EffViT-M** | EfficientViT-B1 | 4.8M | $$\text{Planned}$$ | $$\text{Planned}$$ | |
| **EfficientSAM3-EffViT-L** | EfficientViT-B2 | 15M | $$\text{Planned}$$ | $$\text{Planned}$$ | Highest capacity student model for max accuracy. |

---

## 5. Development To-Do List

- [ ] **Implement Prompt-in-the-Loop Distillation Framework**: Design the training loop for encoder distillation with prompt-guided feature alignment
- [ ] **Integrate Student Backbones**: Add RepViT, TinyViT, and EfficientViT architectures to the codebase
- [ ] **Adapt Perceiver-based Memory Module**: Port EdgeTAM's memory compression technique and integrate with SAM3's architecture
- [ ] **Stage 1 Training**: Execute encoder distillation across all nine student backbones using SAM1 dataset
- [ ] **Stage 2 Training**: Train temporal memory compression module using SAM2 video dataset
- [ ] **Stage 3 Training**: Perform end-to-end fine-tuning on SAM3 official dataset
- [ ] **Benchmark & Evaluation**: Measure inference latency, memory footprint, and segmentation quality on edge devices (mobile, embedded)

---

## License

This project builds upon [SAM](https://github.com/facebookresearch/segment-anything), [SAM2](https://github.com/facebookresearch/sam2), SAM3, [EdgeSAM](https://github.com/chongzhou96/EdgeSAM), [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), [EfficientTAM](https://github.com/yformer/EfficientTAM), [RepViT](https://github.com/THU-MIG/RepViT), [TinyViT](https://github.com/wkcn/TinyViT), and [EfficientViT](https://github.com/mit-han-lab/efficientvit). Please refer to their respective licenses for usage terms.

## Citation

If you use EfficientSAM3 in your research, please cite:

```bibtex
@misc{efficientsam3,
  title={EfficientSAM3: Progressive Hierachical Knowledge Distillation (PhD) from SAM1, 2 and 3},
  author={Zeng, Chengxi Simon and Jiang, Yuxuan and Zhang, Aaron},
  institution={University of Bristol},
  year={2025},
  howpublished={\url{https://github.com/SimonZeng7108/efficientsam3}}
}
```

## Acknowledgments

This work is inspired by and builds upon:
- **[SAM](https://github.com/facebookresearch/segment-anything)** (Meta AI) - Foundation segmentation model
- **[SAM2](https://github.com/facebookresearch/sam2)** - Video object segmentation capabilities
- **SAM3** - Promptable Concept Segmentation
- **[EdgeSAM](https://github.com/chongzhou96/EdgeSAM)** - Efficient encoder distillation techniques
- **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** - Perceiver-based memory compression for tracking
- **[EfficientTAM](https://github.com/yformer/EfficientTAM)** - Efficient temporal attention mechanisms
- **[RepViT](https://github.com/THU-MIG/RepViT)** - Mobile-optimized Vision Transformer backbones
- **[TinyViT](https://github.com/wkcn/TinyViT)** - Tiny Vision Transformer architectures
- **[EfficientViT](https://github.com/mit-han-lab/efficientvit)** - Efficient Vision Transformer models

