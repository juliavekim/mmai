# Homework 2 — Multimodal Fusion and Alignment

## Setup

This notebook runs on Google Colab. Switch to a GPU runtime before running (Runtime → Change runtime type → T4 GPU or equivalent).

**Install dependencies:**
```bash
pip install gdown memory_profiler ftfy regex tqdm
pip install torch==2.3.1 torchvision==0.18.1 torchtext==0.18.0 torchaudio==2.3.1
pip install git+https://github.com/openai/CLIP.git
```

**Clone MultiBench:**
```bash
git clone https://github.com/pliang279/MultiBench.git
cd MultiBench
```

## Datasets

**AV-MNIST**  
Audio-visual digit classification dataset from the [MultiBench](https://arxiv.org/abs/2107.07502) benchmark. Contains paired image (28×28 grayscale) and audio (spectrogram) representations of handwritten digits 0–9. Used here as a controlled benchmark for comparing unimodal and multimodal fusion strategies.

```bash
gdown 1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp
tar -xvzf avmnist.tar.gz
```
Set `data_dir = '/content/MultiBench/avmnist'` in the notebook.

**RAVDESS** (continued from HW1)  
1,440 video clips of 24 actors expressing 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised). Approximately balanced across classes and actors. Used here for contrastive learning between waveform and mel-spectrogram representations of the audio track.

```bash
# mount Google Drive first, then:
unzip /content/drive/MyDrive/ravdess.zip -d /content/
```
Set `dataset_root = '/content/ravdess'` in the notebook.

## Pipeline

**AV-MNIST — unimodal baselines**
1. Load image and audio modalities separately via MultiBench dataloader
2. Train image CNN (2× conv-pool, FC head) and audio CNN (strided conv, FC head) independently
3. Hyperparameter grid search over LR ∈ {1e-3, 5e-4, 1e-4}, dropout ∈ {0.2, 0.3, 0.4}, epochs ∈ {10, 15, 20}

**AV-MNIST — multimodal fusion**
1. Pretrain image and audio encoders (output dim 64 each)
2. Fuse via concatenation (128-dim) → MLP head (256 hidden, 10-class output)
3. Also tested: multiplicative interactions fusion

**RAVDESS — contrastive learning**
1. Load audio clips as paired (waveform, mel-spectrogram) views via torchaudio
2. Train two-modality contrastive model: linear encoder + projector per modality, learnable temperature
3. InfoNCE loss (cross-entropy over N×N similarity matrix, diagonal = correct matches)
4. Post-training: extract embeddings, reduce to 2D via PCA, visualise alignment

## Evaluation

| Task | Metric |
|------|--------|
| AV-MNIST digit classification | Accuracy (val + test) |
| Fusion vs. unimodal comparison | Test accuracy, parameter count, peak memory, runtime |
| Contrastive alignment (RAVDESS) | Contrastive loss, cosine similarity distribution (matched vs. random pairs), PCA visualisation |

Models checkpointed to `best_avmnist_model.pt` (unimodal) and `avmnist_lmf.pt` (fusion) during training.

## Files

```
homework-2/
├── MMAI_HW2.ipynb
├── thoughts.txt
└── README.md 
```
