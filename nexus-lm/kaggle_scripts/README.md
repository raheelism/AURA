# Kaggle runnable scripts

This folder provides `.py` entrypoints so you can run the workflow from a **single Kaggle notebook** using shell commands (no `.ipynb` workflow required inside the repo).

## 1) Setup in one Kaggle notebook cell

```bash
!pip install -q -r /kaggle/working/nexus-lm/requirements.txt
```

## 2) Data preparation

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py data-prep \
  --dataset fineweb_edu \
  --output-dir /kaggle/working/data \
  --n-tokens 1500000000 \
  --verify
```

## 3) Full training

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py train \
  --train-bin /kaggle/input/nexus-data/fineweb_edu_train.bin \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --checkpoint-dir /kaggle/working/checkpoints \
  --max-tokens 1500000000
```

Training progress is printed with step/loss/lr/tokens-per-sec/wall-time, and CSV logs are written to:

`/kaggle/working/checkpoints/train_log.csv`

Resume training from a saved checkpoint:

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py train \
  --train-bin /kaggle/input/nexus-data/fineweb_edu_train.bin \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --checkpoint-dir /kaggle/working/checkpoints \
  --resume-from /kaggle/working/checkpoints/ckpt_tokens_500M.pt \
  --max-tokens 1500000000
```

## 4) Ablation run

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py ablations \
  --train-bin /kaggle/input/nexus-data/fineweb_edu_train.bin \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --ablations baseline aurora_s_only
```

## 5) Evaluation

```bash
!python /kaggle/working/nexus-lm/kaggle_scripts/run.py evaluate \
  --ckpt-path /kaggle/working/checkpoints/ckpt_tokens_1500M.pt \
  --val-bin /kaggle/input/nexus-data/fineweb_edu_val.bin \
  --tokenizer-path /kaggle/input/nexus-data/tokenizer.model \
  --output-json /kaggle/working/eval_metrics.json
```
