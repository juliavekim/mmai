# Homework 1 — Dataset & Multimodal Preprocessing

## Overview

This HW explores multimodal preprocessing for emotion recognition. Working with the Ryerson Audio-Visual Database of Emotional Speech and Song ([RAVDESS](https://zenodo.org/record/1188976)) dataset, consisting of 1,440 video clips of 24 professional actors expressing 8 emotions, we extract visual and audio features and visualise their structure in embedding space.
The dataset was chosen for its clean labelling, balanced class distribution, and natural alignment between modalities: each clip contains synchronised facial expression, speech prosody, and body language, making it a strong testbed for multimodal fusion.

## Modalities

| Modality | Representation | Method |
|----------|---------------|--------|
| Visual | 512-dim embedding | Middle frame → ResNet18 (penultimate layer) |
| Audio | 13-dim MFCC vector | librosa, averaged across time |

**Why these two?** Visual and audio channels carry partially complementary emotional signal — facial expression and vocal tone often agree, but diverge in nuanced cases (e.g., suppressed anger). Combining them is a canonical multimodal setup and maps cleanly onto the redundancy/complementarity distinction in Liang et al. (2022).
**Modalities not used:** Text transcripts exist for RAVDESS but were excluded — the spoken sentences are controlled and semantically neutral by design, so text would add little discriminative signal for emotion. Body keypoints could be extracted but were deemed lower priority for a first pass.

## Preprocessing Pipeline

1. Parse emotion labels from RAVDESS filenames 
2. Extract middle frame per video, avoiding transitional frames
3. Embed frame via pretrained `ResNet18` with classification head removed
4. Extract MFCC features from audio track via `librosa`
5. Concatenate visual and audio features into a joint multimodal representation

## Visualisations

Three families of visualisation were produced:

**t-SNE of embeddings** — applied at perplexities of 5, 10, 30, and 50 across visual-only, audio-only, and multimodal feature spaces. At moderate perplexity, emotionally similar categories (calm/neutral) overlap whilst more distinct emotions (happy/angry) form separable clusters. This suggests the embeddings capture genuine emotion-related structure prior to any classifier training.

**Sample frames** — random middle frames displayed with ground-truth labels, confirming correct preprocessing and label alignment.

**Input statistics** — class counts, video length, audio duration, brightness, and RMS energy distributions. Classes are approximately balanced; video and audio lengths are tightly distributed; brightness and RMS energy show spread, introducing pre-model variability.

## Evaluation Plan

| Metric | Rationale | Limitation |
|--------|-----------|------------|
| Accuracy | Simple overall correctness | Obscures class-specific weakness |
| Macro F1 | Equal weight per class | Less immediately interpretable |
| Confusion matrix | Identifies systematic confusion | Not a scalar summary |

Weighted F1 and balanced accuracy were considered but deprioritised given the approximately balanced class distribution.

## Reading Summary

**Liang et al., "Foundations & Trends in Multimodal Machine Learning" (2022)**

The paper establishes three foundational principles — heterogeneity, connections, and interactions — and six core technical challenges: representation, alignment, reasoning, generation, transference, and quantification. The key distinction between *connections* (properties of the data) and *interactions* (properties of the model) is particularly useful: it gives a concrete test for whether a multimodal architecture is genuinely doing cross-modal integration or reducing to a unimodal shortcut.

## Files

```
homework-1/
├── Copy_of_mmai_HW1.ipynb   # preprocessing pipeline & visualisations
└── MAS_Assignment_1.pdf     # written responses
```
