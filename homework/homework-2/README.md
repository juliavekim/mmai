# Homework 2 — Multimodal Fusion and Alignment

## Overview

This HW covers multimodal fusion, alignment, and contrastive learning across two datasets. The AV-MNIST digit classification task is used to benchmark fusion strategies; RAVDESS is extended with contrastive learning to explore cross-modal alignment between waveform and mel-spectrogram representations.

## Structure

| Problem | Topic | Points |
|---------|-------|--------|
| 1 | Tensors | 5 |
| 2 | Einsum | 5 |
| 3 | Unimodal models (AV-MNIST) | 10 |
| 4 | Multimodal fusion | 10 |
| 5 | Efficiency analysis | 10 |
| 6 | Contrastive learning (RAVDESS) | 30 |
| 7 | Reflection | 10 |

---

## Problem 3: Unimodal Baselines (AV-MNIST)

Two unimodal CNN models trained and evaluated on AV-MNIST digit classification.

**Image model** — best config: LR 1e-3, dropout 0.2, 10 epochs. Val accuracy: 69.66%, test accuracy: 64.62%. Performance plateaued across all hyperparameter settings, suggesting architecture is the binding constraint rather than optimisation.

**Audio model** — best config: LR 5e-4, dropout 0.3, 15 epochs. Val accuracy: 42.12%, test accuracy: 41.40%. Substantially weaker than image; gap reflects inductive bias mismatch (CNN not well-suited to time-frequency spectrogram structure) rather than hyperparameter choice.

---

## Problem 4: Multimodal Fusion

Late fusion of pretrained image and audio encoders. Concatenation and multiplicative interaction fusion both tested. Fusion improves over the weaker (audio) unimodal baseline; the gain over the stronger (image) baseline is more modest, consistent with image being the dominant modality.

---

## Problem 6: Contrastive Learning (RAVDESS)

A two-modality contrastive model trained to align waveform and mel-spectrogram representations of the same audio clips using InfoNCE / cross-entropy over similarity matrices.

**Why cross-entropy?** Each batch of N pairs produces an N×N similarity matrix where the diagonal is correct matches. This is an N-way classification problem per row — cross-entropy with softmax is the natural loss, since it forces the correct pair to score higher than all N-1 negatives simultaneously.

**Results:** Contrastive loss remained relatively high (~3.1–3.4) and training was noisy. Post-alignment PCA shows weak but nonzero alignment — matched pairs slightly closer than random, but cosine similarity distributions overlap substantially. Alignment worked best on high-intensity emotion clips with strong prosodic cues; failed most on neutral/low-intensity clips.

**Likely causes of weak alignment:** RAVDESS is small (~1,440 clips), encoders are single linear layers, and waveform/mel-spectrogram are highly correlated (one is deterministically derived from the other), so the model lacks the independent views that contrastive learning benefits from most.

---

## Reading Summary

**ALBEF — Li et al. (2021):** Contrastive alignment of unimodal encoders should precede cross-attention fusion. Without it, cross-attention computes similarity over incompatible embedding geometries. For RAVDESS, this means: (1) contrastive training to pull matching audio-video pairs together, then (2) cross-attention fusion once representations are comparable.

**Platonic Representation Hypothesis — Huh et al. (2024):** At scale, models trained independently on different modalities tend to converge toward shared representations. Unlikely to apply to RAVDESS — dataset is too small and narrow for spontaneous alignment to emerge. Explicit contrastive alignment remains necessary.

## Files

```
homework-2/
└── MMAI_HW2.ipynb
└── thoughts.txt 
```
