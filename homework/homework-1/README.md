# Homework 1 — Dataset & Multimodal Preprocessing

## Setup

This notebook runs on Google Colab. No GPU required for this assignment.

**Install dependencies:**
```bash
pip install opencv-python librosa torch torchvision
```

**Mount Google Drive and unzip RAVDESS:**
```bash
from google.colab import drive
drive.mount('/content/drive')
unzip /content/drive/MyDrive/ravdess.zip -d /content/
```

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song  
1,440 video clips of 24 professional actors (12 male, 12 female) expressing 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised. Classes are approximately balanced. Clips are similar in length, making fixed-length feature extraction straightforward. Emotion labels are encoded directly in filenames (`modality-vocal_channel-emotion-intensity-statement-repetition-actor.mp4`), parsed from field 3.

Chosen for clean labelling, balanced classes, and natural audio-visual alignment — each clip contains synchronised facial expression and speech prosody, making it a reasonable testbed for multimodal fusion.


## Pipeline

1. Glob all `.mp4` files under `Actor_*/`, parse emotion labels from filenames
2. Extract middle frame per video via OpenCV (avoids transitional frames at clip boundaries)
3. Embed frame through pretrained ResNet18 with classification head removed → 512-dim visual feature vector
4. Extract MFCC features from audio track via librosa, averaged across time → 13-dim audio feature vector
5. Concatenate visual and audio features into a joint 525-dim multimodal representation

**Modalities used:** visual (ResNet18 embeddings) and audio (MFCCs). Text transcripts exist in RAVDESS but were excluded — sentences are semantically neutral by design and add little discriminative signal for emotion.

## Evaluation

Planned metrics for downstream classification (HW2+):

| Metric | Rationale | Limitation |
|--------|-----------|------------|
| Accuracy | Simple overall correctness | Obscures class-specific weakness |
| Macro F1 | Equal weight per class regardless of frequency | Less immediately interpretable |
| Confusion matrix | Shows which emotions are systematically confused | Not a scalar summary |

This assignment focuses on preprocessing and visualisation rather than classification. Three visualisation families were produced: t-SNE of embeddings (perplexity sweep over 5, 10, 30, 50), sample frame grids with ground-truth labels, and input distribution plots (class counts, video length, audio duration, brightness, RMS energy).


## Files

```
homework-1/
├── mmai_HW1.ipynb            # preprocessing pipeline & visualisations
├── mmai_HW1_writeup.pdf      # written responses
└── thoughts.txt              # personal, exploratory notes
```
