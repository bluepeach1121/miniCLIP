# MiniCLIP: 

This repository demonstrates the development of a miniature version of OpenAI's [CLIP](https://github.com/openai/CLIP) model, leveraging a smaller dataset (Flickr30k) and more moderate compute. I incorporate several modern innovations, including:

- **ResNet-34** image encoder, rather than a huge ResNet-50/101 or ViT.
- **Linformer**-based text encoder with Byte-Pair Encoding (BPE) or Hugging Face tokenization.
- **Projection MLP** for aligning the image and text embeddings in a shared space.
- **Dynamic Temperature** module to automatically scale the logits based on batch variance.
- **Sharpness-Aware Minimization** (SAM) for better generalization.
- **Mixed-precision training** with PyTorch’s AMP.

## Table of Contents
1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Key Differences from Original CLIP](#3-key-differences)
4. [Implementation Steps](#4-implementation-steps)
5. [Corrections and Iterations](#5-corrections)
6. [How to Run](#6-how-to-run)
7. [Future Work](#7-future-work)

---

## 1. Overview

OpenAI's CLIP is a powerful  model trained on 400M (image, text) pairs, which is significantly large in both data size and compute. This repository aims to **scale down** those ideas for smaller personal projects using Flickr30k. We still follow the same core principle:

1. Train an image encoder (ResNet-based) and text encoder (Linformer-based) to produce matching embeddings.
2. Compute a contrastive loss on the NxN similarity matrix, pulling matched pairs together and pushing others apart.
3. Dynamically adjust the temperature parameter (learned) that scales these logits.
4. Use advanced optimization (SAM) and mixed-precision to maximize stability and efficiency.

---

## 2. Project Structure

Our suggested repository layout:

```
mini_clip/
├── data/
│   └── dataset.py
├── models/
│   ├── encoders.py
│   ├── tokenizer.py
│   ├── projection_mlp.py
│   ├── dynamic_temperature.py
│   └── sam.py
├── training/
│   └── train.py
├── README.md  <-- (This file)
└── ...
```

1. `dataset.py`: Loads Flickr30k (or any other) image-text pairs.
2. `encoders.py`: ResNet-34 image encoder, Linformer text encoder.
3. `tokenizer.py`: Hugging Face or BPE-based tokenization.
4. `projection_mlp.py`: A two-layer MLP with skip connection.
5. `dynamic_temperature.py`: MLP that computes log-temperature from batch variance.
6. `sam.py`: One-step Sharpness-Aware Minimization (SAM) optimizer.
7. `train.py`: Main training loop, orchestrating everything.

---

## 3. Key Differences From Original CLIP

**Original CLIP**:
- Trained on ~400 million image-text pairs.
- Uses large ResNet-50/101 or Vision Transformers.
- Heavily optimized with giant batch sizes across hundreds of GPUs.
- Pretrained BPE from large text corpora.

**MiniCLIP (This Project)**:
- **Smaller dataset** (Flickr30k) with ~30k images.
- **ResNet-34** instead of ResNet-50 or ViT, fewer parameters.
- **Linformer** text encoder for more efficient attention.
- Introduces a moderate two-layer MLP projection.
- **Dynamic Temperature** module that adjusts temperature each batch using the variance of similarity scores.
- **SAM** for better generalization on smaller data.
- Mixed-precision with AMP for speed on single GPU / Colab.

These changes reduce computational requirements, help with limited data, and provide a simpler codebase for demonstration.

---

## 4. Implementation Steps and Q&A

We walked through each file step by step, asking important technical questions before writing code. Here’s a summary:

1. **`dataset.py`**
   - Confirmed how we store paths to images (`train_root`) and annotations (`train_csv`).
   - Decided on basic transforms like resizing, flipping.
   - Chose to return `(image_tensor, raw_caption_string)` from the dataset.

2. **`encoders.py`**
   - Implemented `ImageEncoder` using a pretrained ResNet-34 with `nn.Identity` replacing the final FC.
   - Created a `LinformerTextEncoder` that reads token IDs, uses positional embeddings, and an optional fallback if Linformer is not installed.
   - Provided a minimal `HFTextEncoder` wrapper to unify Hugging Face tokenization and the text encoder.

3. **`projection_mlp.py`**
   - A modern 2-layer MLP with 4× expansion, skip connection, LayerNorm, and GELU activation.
   - Applies the same projection to both image and text embeddings.

4. **`dynamic_temperature.py`**
   - A small MLP that takes the variance of the NxN similarity matrix as input and outputs `log_temp`.
   - Exponentiate to ensure positivity, used each iteration to scale logits in the contrastive loss.

5. **`sam.py`**
   - Implemented a custom one-step SAM optimizer on top of AdamW.
   - Initially called `unscale_()` in both `first_step` and `second_step`, leading to a future error.
   - **Corrected** by storing the ascent vector in `first_step` and reverting it in `second_step`, unscale only once.

6. **`train.py`**
   - Combined everything in a single training loop.
   - We used the **CLIP-style NxN** symmetrical cross-entropy.
   - Integrated **mixed precision** (`autocast`, `GradScaler`).
   - Incorporated a **validation loop** and checkpoint saving every 10 epochs.
   - Added **`tqdm`** progress bars.

### Key Questions We Asked
- **Data constraints**: Where do images and captions live?
- **Architecture**: How big is the MLP? Which activation?
- **Loss**: NxN symmetrical cross entropy?
- **Temperature**: Static vs. dynamic? We used a learned MLP with variance input.
- **SAM**: One-step, multi-step, or basic? We chose one-step with per-layer or global ρ.
- **Mixed Precision**: Needed special care to avoid double `unscale_()` calls.

---

## 5. Corrections and Iterations

1. **Double Unscale Issue**:
   - The first version of SAM tried to unscale the gradients in both `first_step` and `second_step`. This triggered a PyTorch error.
   - We **fixed** it by unscaling only once during `first_step`, storing the ascent vector, and simply reverting the same vector in `second_step`.

2. **Dimension Matching**:
   - We clarified that both image and text encoders must produce the **same** embedding dimension (e.g., 256) if we want a single projection MLP.

3. **`tqdm` Integration**:
   - We added `tqdm` progress bars for training and validation for better monitoring.

4. **Colab / Single GPU**:
   - This pipeline is optimized for a single GPU environment, unlike the massive scale of the original CLIP.

---

## 6. How to Run

1. **Install Requirements**:
   ```bash
   pip install torch torchvision linformer tqdm transformers
   ```

2. **Update Paths**:
   - In `train.py`, set `train_csv`, `train_root`, `val_csv`, `val_root` to valid paths pointing to your Flickr30k files.

3. **Train**:
   ```bash
   python train.py
   ```
   - This will run 50 epochs, show you a `tqdm` progress bar, and save checkpoints every 10 epochs in `./checkpoints`.

4. **Validation**:
   - If you provide `val_csv` and `val_root`, the script will compute validation loss after each epoch.

5. **Checkpoints**:
   - Check `./checkpoints/` for `.pt` files, each containing the model, optimizer, and dynamic temperature states.

---

## 7. Future Work

- **Few-Shot or Zero-Shot Testing**: Evaluate how well the mini-model generalizes to other tasks or datasets.
- **Better Tokenization**: Expand the simplistic usage of Hugging Face GPT-2 tokenizer or refine BPE merges.
- **More Data**: Incorporate larger or more diverse image-text sets.
- **Alternative Efficient Transformers**: Try Performer, Reformer, or Big Bird in place of Linformer.
- **Deployment**: Provide a small inference script to embed images/text and do retrieval.

