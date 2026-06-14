"""
LoRA 하이퍼파라미터 정밀 sweep — α(scaling)·dropout 효과 분리 (§5.3 향후 과제 수행).

§4.5에서 LoRA는 줄곧 α=2r·dropout=0 관례를 고정했다. 본 드라이버는 rank(r=8)·
주입 모듈(q/v)·step(3,000)·batch(8)·lr(2e-4)를 모두 고정하고 다음 두 축만 따로 바꾼다:

  (1) α sweep   : α ∈ {8, 16, 32, 64}  → 스케일링 α/r ∈ {1, 2, 4, 8}
                  (LoRA 업데이트 (α/r)·BA 의 크기. 사실상 LoRA 경로의 유효 학습률.)
  (2) dropout   : dropout ∈ {0, 0.05, 0.1, 0.2}  (α=16 고정)

다른 실험과 동일하게 3개 시드(11711, 42, 1234)로 반복해 mean±std 와, 기준(α=16/drop=0)
대비 paired 차이를 보고한다. 윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python lora_hparam_sweep.py --use_gpu --max_steps 3000 --seeds 11711 42 1234
"""

import argparse
import sys

try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluation
evaluation.TQDM_DISABLE = True

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from lora_experiments import run_config, seed_everything


# 모두 r=8, q/v, lr=2e-4 고정. (그룹, 이름, alpha, dropout)
CONFIGS = [
  ("alpha",   "a=8 (α/r=1)",    8,  0.0),
  ("alpha",   "a=16 (α/r=2)*",  16, 0.0),   # 기준 (= §4.5.1 r8 q/v)
  ("alpha",   "a=32 (α/r=4)",   32, 0.0),
  ("alpha",   "a=64 (α/r=8)",   64, 0.0),
  ("dropout", "drop=0.05",      16, 0.05),
  ("dropout", "drop=0.10",      16, 0.10),
  ("dropout", "drop=0.20",      16, 0.20),
]
BASE = "a=16 (α/r=2)*"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=3000)
  parser.add_argument("--seeds", type=int, nargs='+', default=[11711, 42, 1234])
  args = parser.parse_args()

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  seed_everything(11711)

  print("Quora train/dev 로드 중...")
  train_raw = load_paraphrase_data(args.para_train)
  dev_raw = load_paraphrase_data(args.para_dev)
  train_ds = ParaphraseDetectionDataset(train_raw, args)
  dev_ds = ParaphraseDetectionDataset(dev_raw, args)
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=train_ds.collate_fn)
  dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)
  print(f"train {len(train_ds):,} / dev {len(dev_ds):,} | LoRA r=8 q/v lr=2e-4 | seeds={args.seeds} | "
        f"{args.max_steps} steps × {len(CONFIGS)} configs × {len(args.seeds)} seeds\n")

  accs = {name: [] for _, name, _, _ in CONFIGS}
  for seed in args.seeds:
    print(f"\n########## seed = {seed} ##########")
    for group, name, alpha, dropout in CONFIGS:
      r = run_config(f"{name} (s{seed})", "lora", train_dl, dev_dl, device, lr=2e-4,
                     rank=8, alpha=alpha, targets=('query', 'value'),
                     dropout=dropout, max_steps=args.max_steps, seed=seed)
      accs[name].append(r['dev_acc'])

  def summarize(title, names):
    print(f"\n========== {title} (dev acc, mean ± std) ==========")
    for name in names:
      a = np.array(accs[name])
      seedstr = " ".join(f"{x:.4f}" for x in a)
      print(f"  {name:<16} mean {a.mean():.4f} ± {a.std(ddof=0):.4f}  "
            f"(min {a.min():.4f}, max {a.max():.4f})  [{seedstr}]")

  base = np.array(accs[BASE])
  summarize("(1) α sweep — r=8 q/v, dropout=0", [n for g, n, _, _ in CONFIGS if g == "alpha"])
  summarize("(2) dropout sweep — r=8 q/v, α=16", [BASE] + [n for g, n, _, _ in CONFIGS if g == "dropout"])

  print(f"\n========== 기준({BASE}) 대비 paired 차이 ==========")
  for group, name, _, _ in CONFIGS:
    if name == BASE:
      continue
    d = np.array(accs[name]) - base
    md, sd = d.mean(), d.std(ddof=0)
    wins = int((d > 0).sum())
    flag = "노이즈 내" if abs(md) < (sd + 1e-9) else ("개선" if md > 0 else "하락")
    print(f"  [{group:<7}] {name:<16}: Δ(mean) = {md:+.4f} ± {sd:.4f} | 승 {wins}/{len(d)} | → {flag}")


if __name__ == "__main__":
  main()
