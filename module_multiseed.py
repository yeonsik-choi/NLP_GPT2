"""
LoRA 주입 모듈 ablation — 다중 시드 검증 (§4.5 모듈 차원의 마지막 단일-시드 보강).

§4.5의 모듈 ablation(q/v → +key → +attn_out → +mlp)은 단일 시드였고, "key 추가 시
오히려 하락" 같은 비단조 결과가 노이즈인지 실제인지 불명확했다. 본 드라이버는 동일
rank(r=8, α=16)·동일 step(3,000)·batch 8에서 모듈 구성만 바꿔 3개 시드로 반복하고,
q/v 기준 대비 각 모듈 추가의 paired 효과를 보고한다.

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python module_multiseed.py --use_gpu --max_steps 3000 --seeds 11711 42 1234
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


# 동일 rank=8, α=16. 모듈 구성만 변화.
CONFIGS = [
  ("qv",        ('query', 'value')),
  ("qkv",       ('query', 'key', 'value')),
  ("qkv+out",   ('query', 'key', 'value', 'attn_out')),
  ("all",       ('query', 'key', 'value', 'attn_out', 'mlp')),
]


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
  print(f"train {len(train_ds):,} / dev {len(dev_ds):,} | LoRA r=8 a=16 | seeds={args.seeds} | "
        f"{args.max_steps} steps × {len(CONFIGS)} 모듈구성 × {len(args.seeds)} seeds\n")

  accs = {name: [] for name, _ in CONFIGS}
  meta = {}
  for seed in args.seeds:
    print(f"\n########## seed = {seed} ##########")
    for name, targets in CONFIGS:
      r = run_config(f"{name} (s{seed})", "lora", train_dl, dev_dl, device, lr=2e-4,
                     rank=8, alpha=16, targets=targets, max_steps=args.max_steps, seed=seed)
      accs[name].append(r['dev_acc'])
      meta[name] = (r['trainable'], r['pct'])

  print("\n\n========== 모듈 ablation 다중 시드 요약 (dev acc, mean ± std) ==========")
  rows = []
  for name, _ in CONFIGS:
    a = np.array(accs[name]); tr, pct = meta[name]
    rows.append((name, tr, pct, a.mean(), a.std(ddof=0), a.min(), a.max(), a))
  for name, tr, pct, m, s, mn, mx, a in rows:  # 모듈 누적 순서 유지(정렬하지 않음).
    seedstr = " ".join(f"{x:.4f}" for x in a)
    print(f"{name:<10} {tr:>10,} ({pct:5.3f}%)  mean {m:.4f} ± {s:.4f}  "
          f"(min {mn:.4f}, max {mx:.4f})  [{seedstr}]")

  # q/v 기준 대비 모듈 추가의 paired 효과.
  base = np.array(accs["qv"])
  print("\n========== q/v 기준 대비 모듈 추가 효과 (paired) ==========")
  for name, _ in CONFIGS:
    if name == "qv":
      continue
    d = np.array(accs[name]) - base
    md, sd = d.mean(), d.std(ddof=0)
    wins = int((d > 0).sum())
    flag = "노이즈 내" if abs(md) < (sd + 1e-9) else ("개선" if md > 0 else "하락")
    print(f"  qv → {name:<8}: Δ(mean) = {md:+.4f} ± {sd:.4f} | {name} 승 {wins}/{len(d)} | → {flag}")


if __name__ == "__main__":
  main()
