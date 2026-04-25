# CNN and U-Net Improvements — Final Report

## Results Summary

| Model | Loss | IoU | Dice | Precision | Recall | F-measure | PR AUC |
|---|---|---|---|---|---|---|---|
| CNN v1 (baseline) | 0.4762 | 0.0674 | 0.1037 | 0.4161 | 0.0763 | 0.1037 | 0.4013 |
| **CNN v2 (improved)** | 0.5495 | 0.4453 | 0.5885 | 0.5296 | 0.7788 | **0.5885** | **0.5820** |
| U-Net v1 (baseline) | 0.4588 | 0.4370 | 0.5870 | 0.5013 | 0.8543 | 0.5870 | 0.6534 |
| **U-Net v2 (improved)** | 0.3586 | 0.5653 | 0.6980 | 0.6429 | 0.8473 | **0.6980** | **0.7725** |

---

## CNN Changes

### What was wrong with CNN v1

The original `CNNAutoencoder` (`src/models/cnn.py`) had three problems:

1. **No skip connections** — all spatial detail was destroyed through aggressive downsampling to a 32×32 bottleneck with no way to recover fine boundaries.
2. **Class imbalance ignored** — plain `BCEWithLogitsLoss` with no weighting caused the model to predict all-background for the first 7 of 10 epochs, since predicting background is always "safe."
3. **Too few epochs, too little capacity** — 10 epochs with max 64 channels was nowhere near enough for the model to learn meaningful segmentation.

---

### Change 1 — Skip Connections + Batch Normalization

**File:** `src/models/cnn_improved.py` — class `CNNWithSkips`

**What skip connections do:**  
In a standard autoencoder, the encoder compresses the image into a small bottleneck (32×32), and the decoder must reconstruct the mask from that alone. Fine spatial details — edges, object boundaries — are lost because there is no direct path for them to flow from encoder to decoder.

Skip connections add a direct "shortcut" from each encoder stage to the corresponding decoder stage. At each resolution level, the decoder receives both the upsampled feature map from the previous decoder stage *and* the original feature map from the matching encoder stage, concatenated together. This means boundary information at 256×256, 128×128, and 64×64 is never permanently discarded — it is passed directly to the decoder at the right resolution.

**What batch normalization does:**  
Without batch normalization, the distribution of activations shifts as the network trains, making each layer's weights fight against changing inputs. Batch normalization normalizes the output of each convolution to have zero mean and unit variance, stabilizing training and allowing the network to converge faster and more reliably.

**Exact architecture change:**

Original (`src/models/cnn.py`):
```
Encoder: Conv→ReLU→Pool (×3, channels: 3→16→32→64)
Decoder: ConvTranspose→ReLU (×3) + Conv1×1
No connection between encoder and decoder stages
```

Improved (`src/models/cnn_improved.py`):
```
Encoder: Conv→BatchNorm→ReLU→Pool (×3, channels: 3→16→32→64)
         ↓ e1 (16ch)   ↓ e2 (32ch)   ↓ e3 (64ch)   ← saved for skip
Bottleneck: pool(e3) → 64ch at 32×32

Decoder stage 3: ConvTranspose(64→32) → concat with e3(64ch) → Conv(96→32)
Decoder stage 2: ConvTranspose(32→16) → concat with e2(32ch) → Conv(48→16)
Decoder stage 1: ConvTranspose(16→8)  → concat with e1(16ch) → Conv(24→8)
Output: Conv1×1(8→1)
```

The `torch.cat([d3, e3], dim=1)` calls at lines 47, 51, 55 in `cnn_improved.py` perform the concatenation. The channel sizes after concatenation (96, 48, 24) are what the subsequent conv blocks take as input.

---

### Change 2 — BCEDice Loss with `pos_weight`

**File:** `src/losses.py` — class `BCEDiceLoss`  
**Used in:** `src/train_cnn_v2.py` lines 111–113

**What BCEDice loss does:**  
The original `BCEWithLogitsLoss` computes pixel-wise cross-entropy. On ECSSD, roughly 65% of pixels are background. A model that predicts all-background achieves ~35% pixel error, so BCE gives it a "comfortable" loss value and the gradient pushes it only weakly toward predicting foreground. This is why the original CNN predicted nothing for 7 epochs.

`BCEDiceLoss` combines two terms:
- **BCE** (`BCEWithLogitsLoss` with `pos_weight=2.0`): the `pos_weight` multiplies the loss on foreground pixels by 2, making the model twice as penalized for missing a foreground pixel as for a false positive. This directly counteracts the class imbalance.
- **Soft Dice**: instead of per-pixel classification, Dice measures the overlap between the predicted probability map and the ground truth mask across the whole image: `2 * (pred · target) / (pred + target)`. Dice loss is 0 when the prediction perfectly overlaps the mask and 1 when there is no overlap. Because it operates on the full mask at once, it is naturally balanced regardless of class frequency.

The final loss is `0.5 * BCE + 0.5 * Dice`, so both objectives contribute equally.

```python
# src/losses.py
intersection = (probs * targets).sum(dim=(1, 2, 3))
union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss.mean()
```

**Effect on CNN:** Recall jumped from 0.0763 → 0.7788. The model stopped predicting all-background immediately from epoch 1.

---

### Change 3 — Data Augmentation

**File:** `src/dataset.py` — class `AugmentedECSSDDataset`  
**Used in:** `src/train_cnn_v2.py` line 101 (`augment=True` for train only)

**What augmentation does:**  
With only 700 training images, the model sees the exact same pixels every epoch. Augmentation synthetically increases dataset variety by applying random transformations, forcing the model to generalise to different orientations, lighting conditions, and positions rather than memorising training images.

Three augmentations are applied, in order, only to the training set:

1. **Horizontal flip (p=0.5):** Mirrors the image and mask left-right with 50% probability. Salient objects appear equally often on the left and right side of images, so this is always valid for segmentation.  
   `TF.hflip(image)` / `TF.hflip(mask)` — applied to both identically.

2. **Random rotation ±15°:** Rotates the image and mask by a random angle drawn uniformly from [−15°, 15°]. Small enough not to cut off objects, but teaches the model that salient objects can be tilted.  
   `TF.rotate(image, angle)` / `TF.rotate(mask, angle)` — applied identically.

3. **Color jitter (image only):** Randomly adjusts brightness (±30%), contrast (±30%), and saturation (±20%). Applied only to the image, *not* the mask, since mask labels are not affected by colour. Teaches the model to focus on shape, not specific colour values.  
   `transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(image)`

Validation and test sets use `augment=False` and remain fully deterministic.

---

### Change 4 — Learning Rate and Epochs

**File:** `src/train_cnn_v2.py` lines 74–76, 122–126

**Original:** LR = 1e-3, 10 epochs, no scheduler.

**Improved:** LR = 3e-4, 100 epochs, `ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)`.

**Why lower LR:** At 1e-3, the first run (50 epochs) oscillated on the validation Dice for 40 epochs before the scheduler finally stepped in. Starting at 3e-4 reaches stable convergence much sooner — the model behaves consistently from epoch 1.

**What `ReduceLROnPlateau` does:** After every epoch, the scheduler checks whether the validation Dice improved. If it did not improve for `patience=5` consecutive epochs, the learning rate is multiplied by `factor=0.5` (halved). This allows the model to take large steps early in training and fine-tune with small steps once it approaches a good solution, without needing to manually set a schedule. The LR halved repeatedly: 3e-4 → 1.5e-4 → 7.5e-5 → 3.75e-5 → 1.87e-5 → 1e-5 (min), reflecting the model progressively converging.

---

## U-Net Changes

### What was wrong with U-Net v1

U-Net v1 had good recall (0.8543) but poor precision (0.5013). This means the model successfully detected most salient objects, but also incorrectly labelled large areas of background as foreground — over-segmentation. Two causes:

1. **BCE loss alone** does not penalise false positives strongly enough. Predicting foreground everywhere keeps recall high and loss acceptable.
2. **No regularization** — the 1024-channel bottleneck, being the largest layer, would fit spurious background patterns on the small training set.

---

### Change 1 — Dropout in Bottleneck

**File:** `src/models/unet_improved.py` — class `UNetImproved`, lines 47–50

**What spatial dropout does:**  
`nn.Dropout2d(p=0.3)` zeros entire feature *channels* (not individual neurons) with 30% probability during training. Applied after the bottleneck `DoubleConv`, it prevents any single bottleneck channel from becoming dominant. This forces the decoder to rely more heavily on the skip connections (which carry precise spatial information from the encoder) rather than on broad, potentially noisy, global features from the bottleneck.

The practical effect is reduced over-segmentation: the decoder must use local boundary cues from skip connections to decide where the object ends, rather than just propagating a confident "foreground everywhere" signal from the bottleneck.

```python
# src/models/unet_improved.py
self.bottleneck = nn.Sequential(
    DoubleConv(features[-1], features[-1] * 2),
    nn.Dropout2d(p=0.3),      # zeroes entire channels, not pixels
)
```

**Effect:** Precision jumped from 0.5013 → 0.6429 (+0.14) while recall stayed stable at 0.8473 (barely changed from 0.8543). Dropout achieved exactly its intended purpose.

---

### Change 2 — BCEDice Loss (no `pos_weight`)

**File:** `src/losses.py` — class `BCEDiceLoss`  
**Used in:** `src/train_unet_v2.py` line 109 (`BCEDiceLoss(bce_weight=0.5)`, no `pos_weight`)

Same `BCEDiceLoss` as CNN, but without `pos_weight` for U-Net. The Dice component already handles class imbalance naturally (it normalises over predicted vs. actual foreground), and U-Net's skip connections give it enough spatial information to find foreground without needing the extra nudge. Adding `pos_weight` to U-Net would push recall even higher at the expense of precision, worsening the over-segmentation problem.

**Effect:** Directly optimising the Dice coefficient (which is also the validation metric used for model selection) aligns the training objective with evaluation, producing more consistent improvements epoch to epoch. Compare val Dice progression: U-Net v1 fluctuated wildly (0.44→0.57→0.45→0.57...) while U-Net v2 trended steadily upward.

---

### Change 3 — Data Augmentation

**File:** `src/dataset.py` — class `AugmentedECSSDDataset`  
**Used in:** `src/train_unet_v2.py` line 101 (`augment=True` for train only)

Same three augmentations as CNN (horizontal flip, ±15° rotation, color jitter). For U-Net, augmentation primarily helps precision: it prevents the high-capacity model (7M+ parameters) from memorising background textures in the 700 training images, which was contributing to false positives.

---

### Change 4 — Epochs

**File:** `src/train_unet_v2.py` line 74

**Original:** 10 epochs.  
**Improved:** 100 epochs (initially run for 50, extended to 100).

U-Net v1 stopped at epoch 10 while still actively converging — train loss was 0.36 and dropping. U-Net v2's scheduler never even fired during the 50-epoch run (LR stayed at 1e-3 throughout), confirming the model was still in an active learning phase. Extending to 100 epochs allows the scheduler to step in naturally and fine-tune the model to its full potential.

---

## What Did Not Change

- **Model selection criterion:** Best model saved on highest validation Dice, same as v1.
- **Optimizer:** Adam with the same base settings — only LR changed.
- **Data split:** 70/15/15 train/val/test, fixed seed 42 — test set is identical to v1 for fair comparison.
- **Evaluation metrics:** IoU, Dice, Precision, Recall, F-measure, PR AUC — all unchanged.
- **U-Net encoder/decoder depth:** Features [64, 128, 256, 512] unchanged — only the bottleneck was modified.
