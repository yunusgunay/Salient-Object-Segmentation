# From Scratch to Pretrained: Salient Object Segmentation

This project investigates the performance of various deep learning architectures—ranging from simple from-scratch models to advanced pretrained Vision Transformers - on the task of identifying visually prominent objects in complex natural scenes. Using the Extended Complex Scene Saliency Dataset (ECSSD), we implemented and iteratively improved four major approaches to understand how architectural choices and training methodologies impact segmentation quality.

## Project Overview

Salient object segmentation is fundamentally more challenging than image classification because it requires pixel-level binary decisions, demanding both high-level semantic understanding and fine-grained spatial reasoning. We tested the hypothesis that pretrained encoders provide semantically richer representations that outperform from-scratch models on small datasets, and that targeted architectural improvements can close this performance gap.

## Architectures and Learning Journey

Our experiments were conducted in two rounds: a **v1 Baseline** to establish initial performance and a **v2 Improved** round where we applied specific architectural and training enhancements.

### From-Scratch Models

* **CNN-Based Autoencoder**:
* **v1 Baseline**: A simple three-stage encoder-decoder. It suffered from "class imbalance collapse," where the model only predicted background pixels.

* **v2 Improvements**: We implemented **skip connections** to pass high-resolution spatial detail directly to the decoder. We also added **batch normalization** for stability and switched to **BCEDice loss** to handle class imbalance.

* **Result**: Performance improved from an F-measure of 0.10 to 0.59.

* **U-Net**:
* **v1 Baseline**: Already utilized skip connections, yielding strong recall (0.85) from the first epoch but suffering from over-segmentation.
* **v2 Improvements**: We added **Spatial Dropout (p=0.3)** after the bottleneck to regularize the model and prevent it from memorizing the small training set.
* **Result**: Achieved a more stable and generalizable F-measure of 0.79.

### Pretrained Vision Transformers
We utilized frozen encoders to prevent overfitting on our small dataset (700 training images).
* **DINO**: Used a ViT-Small encoder pretrained via self-supervised learning. It provided the most spatially precise features.
* **CLIP**: Used a ViT-Base encoder pretrained via image-text contrastive learning. While semantically strong, it was initially less spatially precise than DINO.
* **Late Fusion**: We combined DINO and CLIP by averaging their output logits. This approach achieved the best overall performance with an F-measure of 0.90.

## Key Results
| Model | Round | F-measure | PR AUC |
| --- | --- | --- | --- |
| CNN | v1 (150 epochs) | 0.4097 | 0.4711 |
| CNN | v2 (100 epochs) | 0.5885 | 0.5820 |
| U-Net | v2 (150 epochs) | 0.7976 | 0.8770 |
| DINO | v2 (Improved) | 0.8838 | 0.9597 |
| CLIP | v2 (Improved) | 0.8660 | 0.9516 |
| **DINO + CLIP** | **Fusion** | **0.8959** | **0.9720** |

## Core Learning Insights
1. **Architecture vs. Training Time**: Extending training for the CNN baseline to 150 epochs improved results, but it could not overcome the architectural ceiling caused by the lack of skip connections.
2. **Loss Function Matters**: Moving from standard Binary Cross-Entropy (BCE) to a hybrid **BCEDice loss** was critical for training on the class-imbalanced ECSSD dataset, as it forced the models to optimize for spatial overlap.
3. **Regularization for Small Data**: In U-Net, spatial dropout was essential to move from memorization to generalization, significantly improving precision by reducing false-positive foreground predictions.
4. **Pretrained Power**: Pretrained encoders (DINO/CLIP) reached high performance almost immediately, demonstrating that semantically rich features are more valuable than complex decoders when data is limited.

## Environment and Tools
* **Framework**: Python, PyTorch.
* **Libraries**: `timm` (for ViT encoders), `torchvision`, `scikit-learn`, `matplotlib`.
* 
**Hardware**: Trained on Google Colab using NVIDIA GPU acceleration.
