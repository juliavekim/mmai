
# Homework 3 — Testing and Fine-Tuning VLMs

## Setup

Requires an A100 GPU. In Colab: Runtime → Change runtime type → A100. 

**Install dependencies:**
```bash
pip install transformers accelerate bitsandbytes pillow torch peft -q
```

## Dataset

**CIFAR-10**  
60,000 32×32 colour images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 6,000 images per class. Downloaded via torchvision. Converted into a VLM question-answer format: each image is paired with the question `"What object is shown in this image?"` and the ground-truth class label as the answer.

Dataset prepared as:
```
mmai-data/
├── images/
│   ├── image_01.jpg
│   └── ...
└── data.jsonl        # {"image": "images/1.jpg", "question": "...", "answer": "..."}
```

Upload to Google Drive as a zip, then mount and unzip in Colab. Train/test split: 80/20, random sampling to preserve class balance. The 4 held-out test images (`image_10`, `image_25`, `image_60`, `image_120`) are excluded from training and prompt development entirely.

## Pipeline

**Baseline inference**  
Load pretrained `Qwen2.5-VL-3B-Instruct` via HuggingFace. Run on 4 held-out CIFAR-10 images with varying system prompts: plain assistant, label-constrained, few-shot, and chain-of-thought reasoning.

**LoRA fine-tuning**  
Fine-tune Qwen2.5-VL-3B-Instruct using LoRA (via `peft`) with a grid search over:
- Learning rate: {1e-3, 1e-4}
- LoRA rank r: {4, 8}
- Epochs: {3, 5}

Best config: LR 1e-4, r=4, alpha=8, 3 epochs, batch size 1, gradient accumulation 1, image shortest edge 288px, max sequence length 512. Checkpoints saved to `/content/qwen_lora_grid/`. Best model selected by lowest final training loss.

**Post-training evaluation**  
Reload best checkpoint via `PeftModel.from_pretrained`, run on the same 4 held-out images, compare against pretrained baseline.

## Evaluation

| Stage | Metric |
|-------|--------|
| Baseline inference | Correctness on 4 held-out images, qualitative failure analysis |
| Prompt engineering | Accuracy across 4 prompt strategies |
| LoRA fine-tuning | Training loss per config |
| Post-training | Correct predictions on held-out set vs. pretrained baseline |

**Results summary:** Pretrained model: 1/4 correct (out-of-vocabulary labels for most classes). Label-constrained prompting: 2/4. LoRA fine-tuned: 3/4, with all OOV errors corrected.


## Files

```
homework-3/
├── mmai_HW3.ipynb
├── mmai_HW3_writeup.pdf
├── thoughts.txt
└── README.md
```
