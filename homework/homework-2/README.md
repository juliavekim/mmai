# Homework 2 — Multimodal Fusion and Alignment

## Overview

This HW extends HW1 into fusion and alignment. AV-MNIST is used as a clean benchmark for comparing unimodal vs. multimodal fusion strategies; RAVDESS is extended with contrastive learning to explore cross-modal alignment between waveform and mel-spectrogram representations.

## Unimodal Baselines (AV-MNIST)

Two unimodal CNN models trained on AV-MNIST digit classification. Image substantially outperforms audio (~64.6% vs ~41.4% test accuracy). The gap reflects inductive bias mismatch: a standard spatial CNN isn't well-suited to the time-frequency structure of spectrograms — rather than anything fixable by hyperparameter tuning. Plateau in image model performance across all configs suggests architecture is the binding constraint.

## Multimodal Fusion

Late fusion of pretrained image and audio encoders via concatenation and multiplicative interactions. Fusion improves over the weaker audio baseline; gains over the image baseline are modest, consistent with image being the dominant modality. Efficiency analysis confirms the expected tradeoff: more parameters and memory for marginal accuracy gains, which matters in practice depending on deployment constraints.


## Contrastive Learning (RAVDESS)

A two-modality contrastive model trained to align waveform and mel-spectrogram representations of the same clips using InfoNCE. Loss remained high (~3.1–3.4) throughout training. Post-alignment PCA shows weak but nonzero alignment: matched pairs slightly closer than random, cosine similarity distributions overlap substantially. Alignment worked best on high-intensity emotion clips; failed most on neutral/low-intensity ones.

Waveform + mel-spectrogram is probably the wrong modality pair for contrastive learning: one is deterministically derived from the other, so there are no genuinely independent views for the model to reconcile. Small batch size (N=32, so 31 negatives per anchor) compounds this; contrastive learning benefits from many negatives. These are structural issues, not hyperparameter ones.

## Reading Summary

**ALBEF (Li et al., 2021):** Unimodal contrastive alignment should precede cross-attention fusion — without it, attention scores are computed over geometrically incompatible spaces. For RAVDESS: align audio and video encoders contrastively first, then fuse.

**Platonic Representation Hypothesis (Huh et al., 2024):** At scale and with diverse data, independently trained models tend to converge toward shared representations. Unlikely to apply to RAVDESS — 1,440 clips is nowhere near the regime where spontaneous alignment has been observed. Explicit alignment remains necessary.

## Files

```
homework-2/
└── MMAI_HW2.ipynb
└── thoughts.txt
```
