"""
다중 시드 PEFT 실험 드라이버 (Paraphrase) — 통계적 유의성 보강.

기존 lora_experiments.py 의 ablation 은 단일 시드(11711)였다. 보고서 §5.2 가
인정한 가장 큰 한계 — "mid-rank(r8 vs r16) 차이가 노이즈 범위인지 실제인지
단일 시드라 알 수 없다" — 를 메우기 위해, 헤드라인 주장을 좌우하는 설정들을
여러 시드로 반복 학습하여 평균±표준편차와 시드별 승패(paired)를 보고한다.

검증 대상 주장:
  (1) "LoRA r=32(q/v) ≈ full fine-tuning" 이 노이즈 내 동급인가.
  (2) "용량은 모듈 확대보다 rank 확대" — rank 사다리(r4→r32)가 단조 증가하는가,
      mid-rank(r8 vs r16) 차이는 시드 분산 안에 들어가는가.
  (3) "ReFT 는 절반 파라미터로 동급 LoRA(r8 q/v)에 근접" 이 시드에 견고한가.

공정 비교를 위해 모든 설정은 동일 step(--max_steps), 동일 batch, 동일 시드 집합으로
학습한다. 윤리: train/dev 만 사용, 공식 test 미사용.

실행:
  python multiseed_experiments.py --use_gpu --max_steps 3000 --seeds 11711 42 1234
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
evaluation.TQDM_DISABLE = True  # 다중 run 로그가 진행바로 부푸는 것 방지.

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from lora_experiments import run_config, seed_everything


# 헤드라인 주장을 좌우하는 설정들. (이름, run_config 키워드 인자)
CONFIGS = [
  ("full(ref)",   dict(method="full", lr=1e-5)),
  ("lora r=4 qv", dict(method="lora", lr=2e-4, rank=4,  alpha=8,  targets=('query', 'value'))),
  ("lora r=8 qv", dict(method="lora", lr=2e-4, rank=8,  alpha=16, targets=('query', 'value'))),
  ("lora r=16 qv", dict(method="lora", lr=2e-4, rank=16, alpha=32, targets=('query', 'value'))),
  ("lora r=32 qv", dict(method="lora", lr=2e-4, rank=32, alpha=64, targets=('query', 'value'))),
  ("reft r=8",    dict(method="reft", lr=2e-4, reft_rank=8)),
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
  print(f"train {len(train_ds):,} / dev {len(dev_ds):,} | seeds={args.seeds} | "
        f"{args.max_steps} steps × {len(CONFIGS)} configs × {len(args.seeds)} seeds\n")

  # accs[name] = [seed별 dev acc], trainable/pct 는 시드 무관(설정 고정)이라 마지막 값 저장.
  accs = {name: [] for name, _ in CONFIGS}
  meta = {}
  for seed in args.seeds:
    print(f"\n########## seed = {seed} ##########")
    for name, cfg in CONFIGS:
      r = run_config(f"{name} (s{seed})", train_dl=train_dl, dev_dl=dev_dl,
                     device=device, max_steps=args.max_steps, seed=seed, **cfg)
      accs[name].append(r['dev_acc'])
      meta[name] = (r['trainable'], r['pct'], r['its'])

  # ── 요약: 평균±표준편차 (평균 내림차순) ──
  print("\n\n========== 다중 시드 요약 (dev accuracy, mean ± std) ==========")
  rows = []
  for name, _ in CONFIGS:
    a = np.array(accs[name])
    tr, pct, its = meta[name]
    rows.append((name, tr, pct, its, a.mean(), a.std(ddof=0), a.min(), a.max(), a))
  rows.sort(key=lambda x: -x[4])
  print(f"{'설정':<14} {'학습파라미터':>13} {'비율%':>7} {'it/s':>6} "
        f"{'mean':>7} {'± std':>7} {'min':>7} {'max':>7}   시드별")
  for name, tr, pct, its, m, s, mn, mx, a in rows:
    seedstr = " ".join(f"{x:.4f}" for x in a)
    print(f"{name:<14} {tr:>13,} {pct:>7.3f} {its:>6.1f} "
          f"{m:>7.4f} {s:>7.4f} {mn:>7.4f} {mx:>7.4f}   [{seedstr}]")

  # ── paired 비교: 동일 시드 집합이므로 시드별 차이로 검정 ──
  def paired(a_name, b_name):
    a = np.array(accs[a_name]); b = np.array(accs[b_name])
    d = a - b
    wins = int((d > 0).sum())
    md, sd = d.mean(), d.std(ddof=0)
    # 시드 수가 적으므로 정규성 가정 대신 평균차±std 와 승패를 함께 보고.
    flag = "동급(노이즈 내)" if abs(md) < (sd + 1e-9) else f"{a_name} 우세" if md > 0 else f"{b_name} 우세"
    print(f"  {a_name:<13} vs {b_name:<13}: Δ(mean) = {md:+.4f} ± {sd:.4f} | "
          f"{a_name} 승 {wins}/{len(d)} | → {flag}")

  print("\n========== 헤드라인 주장 paired 검정 ==========")
  print("[주장1] LoRA r=32(q/v) 가 full FT 와 동급인가")
  paired("lora r=32 qv", "full(ref)")
  print("[주장2] rank 확대 효과 (사다리) + mid-rank 차이가 노이즈인가")
  paired("lora r=32 qv", "lora r=4 qv")
  paired("lora r=16 qv", "lora r=8 qv")
  print("[주장3] ReFT r=8 가 동급 파라미터대 LoRA r=8(q/v) 에 근접한가")
  paired("lora r=8 qv", "reft r=8")


if __name__ == "__main__":
  main()
