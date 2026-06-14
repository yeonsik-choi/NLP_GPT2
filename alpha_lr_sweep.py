"""
α × lr 2D sweep — "α는 그냥 lr를 키운 것인가?"를 분리 검증 (Tier 1 #1).

§4.5.3은 α(scaling)를 키우면 정확도가 오른다고 보였으나, LoRA 업데이트가
(α/r)·lr·∇ 형태라 α 증가가 lr 증가와 구분되는지는 검증하지 않았다. 여기서
r=8·q/v·3,000 step·batch 8을 고정하고 α∈{16,64} × lr∈{1e-4,2e-4,4e-4} 격자를
3개 시드로 돌린다.

핵심 비교(iso-product 대각선): LoRA 경로의 유효 업데이트 ∝ (α/r)·lr 이므로
  α=16/lr=4e-4  (유효 ∝ 2×4e-4 = 8e-4)
  α=64/lr=1e-4  (유효 ∝ 8×1e-4 = 8e-4)
가 같다면 α와 lr은 LoRA 경로에서 곱으로만 작용(중복)함을 뜻한다.

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python alpha_lr_sweep.py --use_gpu --max_steps 3000 --seeds 11711 42 1234
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

ALPHAS = [16, 64]
LRS = [1e-4, 2e-4, 4e-4]


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
  print(f"train {len(train_ds):,} / dev {len(dev_ds):,} | LoRA r=8 q/v | seeds={args.seeds} | "
        f"{args.max_steps} steps × {len(ALPHAS)*len(LRS)} (α×lr) × {len(args.seeds)} seeds\n")

  accs = {}  # (alpha, lr) -> [seed accs]
  for seed in args.seeds:
    print(f"\n########## seed = {seed} ##########")
    for alpha in ALPHAS:
      for lr in LRS:
        r = run_config(f"a={alpha} lr={lr:.0e} (s{seed})", "lora", train_dl, dev_dl, device, lr=lr,
                       rank=8, alpha=alpha, targets=('query', 'value'), max_steps=args.max_steps, seed=seed)
        accs.setdefault((alpha, lr), []).append(r['dev_acc'])

  print("\n\n========== α × lr 격자 (dev acc, mean ± std) ==========")
  header = "α \\ lr   " + "".join(f"{lr:>14.0e}" for lr in LRS)
  print(header)
  for alpha in ALPHAS:
    row = f"α={alpha:<6}"
    for lr in LRS:
      a = np.array(accs[(alpha, lr)])
      row += f"  {a.mean():.4f}±{a.std(ddof=0):.4f}"
    print(row)

  print("\n시드별 상세:")
  for alpha in ALPHAS:
    for lr in LRS:
      a = np.array(accs[(alpha, lr)])
      seedstr = " ".join(f"{x:.4f}" for x in a)
      print(f"  α={alpha:<3} lr={lr:.0e}: mean {a.mean():.4f} ± {a.std(ddof=0):.4f}  (유효 α/r·lr ∝ {(alpha/8)*lr:.1e})  [{seedstr}]")

  # iso-product 대각선 비교: α16/lr4e-4 vs α64/lr1e-4 (둘 다 유효 ∝ 8e-4).
  print("\n========== iso-product 비교 (유효 업데이트 ∝ 8e-4) ==========")
  a16 = np.array(accs[(16, 4e-4)]); a64 = np.array(accs[(64, 1e-4)])
  d = a64 - a16
  print(f"  α=16/lr=4e-4 : {a16.mean():.4f} ± {a16.std(ddof=0):.4f}")
  print(f"  α=64/lr=1e-4 : {a64.mean():.4f} ± {a64.std(ddof=0):.4f}")
  flag = "동급(α≈lr 곱 중복)" if abs(d.mean()) < (d.std(ddof=0) + 1e-9) else ("α=64 우세" if d.mean() > 0 else "α=16 우세")
  print(f"  Δ(mean) = {d.mean():+.4f} ± {d.std(ddof=0):.4f} | → {flag}")


if __name__ == "__main__":
  main()
