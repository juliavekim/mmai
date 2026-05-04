# Homework 4 — Reinforcement Learning for Vision-Language Models (GRPO)

## Setup

Requires an A100 GPU. In Colab: Runtime → Change runtime type → A100.

**Install dependencies:**
```bash
pip install transformers accelerate bitsandbytes pillow torch torchvision trl peft datasets gdown qwen-vl-utils -q
```

## Dataset

**CIFAR-10** (reused from HW3)
60,000 32×32 colour images across 10 classes. Converted to GRPO format: each image is paired with a question and ground-truth label, with an instruction suffix prompting chain-of-thought reasoning before the final `Answer:` token.

```
mmai-data/
├── images/
│   ├── image_01.jpg
│   └── ...
└── data.jsonl        # {"image": "images/1.jpg", "question": "...", "answer": "..."}
```

Instruction suffix:
```
First, think through your reasoning step by step.
Then, provide your final answer as a single word on the last line after "Answer:".
```

Upload to Google Drive as a zip, mount and unzip in Colab. 4 held-out test images excluded from training entirely.

## Algorithm: GRPO

**Group Relative Policy Optimization** eliminates the value model (critic) required by PPO by computing advantages relative to other completions in the same group. For each prompt, G completions are sampled; the advantage of each completion is its reward minus the group mean, divided by the group standard deviation.

$$A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}$$

Two reward functions:
- **Accuracy reward** (binary): 1.0 if extracted answer matches ground truth, 0.0 otherwise
- **Format reward** (binary): 1.0 if response contains `Answer:` marker, 0.0 otherwise

## Pipeline

**Problem 4: GRPO advantage computation**
Implemented `compute_grpo_advantage` — groups rewards by `group_id`, computes per-group mean and std, normalizes, then broadcasts per-sample advantages across the response mask to produce per-token advantages.

**Problem 5: Reward functions**
Implemented `extract_answer` (regex match after `Answer:`), `accuracy_reward` (exact match), and `format_reward` (presence of `Answer:` marker).

**Problem 6: Dataset construction**
Built HuggingFace `Dataset` with `prompt`, `image`, and `answer` columns from `data.jsonl`. Prompt formatted as user/assistant conversation with instruction suffix.

**Problem 7: GRPO training**
Fine-tuned `Qwen/Qwen3-VL-2B-Instruct` using TRL's `GRPOTrainer` with LoRA.

| Hyperparameter | Value |
|---|---|
| Model | Qwen/Qwen3-VL-2B-Instruct |
| Completions per prompt (G) | 2 |
| Max completion length | 256 tokens |
| Learning rate | 1e-5 |
| Max steps | 100 |
| Batch size | 1 |
| Epsilon (clipping) | 0.2 |
| Temperature | 1.2 |
| Beta (KL penalty) | 0.0 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | q_proj, v_proj, k_proj, o_proj |

**Problem 8: Post-training evaluation**
Reloaded LoRA adapters via `PeftModel.from_pretrained`, ran on 3 held-out COCO images (cat, truck, bear).

## Evaluation

| Stage | Metric |
|---|---|
| Reward tracking | Total reward, accuracy reward, format reward per step |
| Format consistency | Fraction of responses containing `Answer:` marker |
| Post-training | Correct predictions on held-out set vs. zero-shot baseline |

Comparison with HW3 (SFT/LoRA):
- GRPO converges more slowly (reward-based signal is noisier than cross-entropy loss)
- GRPO produces more consistent `Answer:` formatting after training (format reward drives this)
- SFT produced faster initial improvement; GRPO shows more gradual but stable reward growth

## Files

```
homework-4/
├── hw4-mmai.ipynb
├── hw4-mmai_writeup.pdf
├── thoughts.txt
└── README.md
```
