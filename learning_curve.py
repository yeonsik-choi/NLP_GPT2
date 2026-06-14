"""
학습 곡선 (dev acc vs step) — 적응 동역학 시각화 (Tier 2 #5).

지금까지의 결과는 모두 종점(endpoint) 수치였다. 여기서는 full FT / LoRA r=32 /
LoRA α=64(r8)의 dev accuracy를 학습 step에 따라 추적해 "누가 더 빨리 적응하는가"를
본다. 매 평가는 비용을 줄이기 위해 dev 앞 2,000개 부분집합으로 측정한다(동일 부분집합).

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python learning_curve.py --use_gpu --max_steps 3000 --eval_every 250
"""

import argparse
import sys

try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import evaluation
evaluation.TQDM_DISABLE = True

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase
from optimizer import AdamW
from lora_experiments import PEFTParaphrase, seed_everything

CONFIGS = [
  ("full",    dict(method="full"),                                                 1e-5),
  ("r32",     dict(method="lora", rank=32, alpha=64, targets=('query', 'value')),  2e-4),
  ("alpha64", dict(method="lora", rank=8,  alpha=64, targets=('query', 'value')),  2e-4),
]


def curve(name, kwargs, lr, train_dl, dev_dl, device, max_steps, eval_every, seed=11711):
  seed_everything(seed)
  model = PEFTParaphrase(**kwargs).to(device)
  optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
  pts = []
  step = 0
  done = False
  while not done:
    for batch in train_dl:
      model.train()
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      loss = F.cross_entropy(model(b_ids, b_mask), labels)
      loss.backward()
      optimizer.step()
      step += 1
      if step % eval_every == 0 or step >= max_steps:
        model.eval()
        acc, *_ = model_eval_paraphrase(dev_dl, model, device)
        pts.append((step, acc))
        print(f"  [{name:<8}] step {step:>4} | dev(2k) acc = {acc:.4f}", flush=True)
      if step >= max_steps:
        done = True
        break
  del model, optimizer
  torch.cuda.empty_cache()
  return pts


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=3000)
  parser.add_argument("--eval_every", type=int, default=250)
  parser.add_argument("--dev_subset", type=int, default=2000)
  args = parser.parse_args()

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  seed_everything(11711)

  print("Quora train/dev 로드 중...")
  train_raw = load_paraphrase_data(args.para_train)
  dev_raw = load_paraphrase_data(args.para_dev)
  train_ds = ParaphraseDetectionDataset(train_raw, args)
  dev_ds = ParaphraseDetectionDataset(dev_raw, args)
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=train_ds.collate_fn)
  dev_sub = Subset(dev_ds, list(range(min(args.dev_subset, len(dev_ds)))))
  dev_dl = DataLoader(dev_sub, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)
  print(f"train {len(train_ds):,} / dev_subset {len(dev_sub):,} | eval every {args.eval_every} steps\n")

  curves = {}
  for name, kwargs, lr in CONFIGS:
    print(f"=== {name} ===")
    curves[name] = curve(name, kwargs, lr, train_dl, dev_dl, device, args.max_steps, args.eval_every)

  steps = [s for s, _ in curves[CONFIGS[0][0]]]
  print("\n\n========== 학습 곡선 (dev 2k acc) ==========")
  print(f"{'step':>6}" + "".join(f"{n:>12}" for n, _, _ in CONFIGS))
  for i, s in enumerate(steps):
    row = f"{s:>6}"
    for n, _, _ in CONFIGS:
      row += f"{curves[n][i][1]:>12.4f}"
    print(row)


if __name__ == "__main__":
  main()
