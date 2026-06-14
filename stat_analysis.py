"""
통계적 유의성 격상 — 예제 단위 McNemar 검정 (Tier 1 #3).

§4.5의 다중 시드 paired 판정은 "평균차±표준편차" 휴리스틱이고 시드 n=3이라
검정력이 약하다. 여기서는 dev 40,429개 예제에 대해 두 모델의 정답/오답을 짝지어
McNemar 검정을 수행한다(예제 단위 n≈40K → 강력). 동일 시드(11711)·동일 step으로
핵심 설정을 학습해 예제별 예측을 모으고, 관심 비교쌍의 p-value를 보고한다.

McNemar: 두 모델의 불일치(discordant) 예제만 본다.
  n10 = A만 맞음, n01 = B만 맞음. 귀무가설: n10 == n01.
  통계량 = (|n10-n01|-1)^2 / (n10+n01)  ~ χ²(1)  (연속성 보정).
  불일치 수가 적으면 정확 이항검정으로 대체.

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python stat_analysis.py --use_gpu --max_steps 3000
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
from scipy.stats import chi2, binomtest
from torch.utils.data import DataLoader

import evaluation
evaluation.TQDM_DISABLE = True

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase
from optimizer import AdamW
from lora_experiments import PEFTParaphrase, seed_everything


# 핵심 설정: (이름, method kwargs, lr)
CONFIGS = [
  ("full",     dict(method="full"),                                                        1e-5),
  ("r8",       dict(method="lora", rank=8,  alpha=16, targets=('query', 'value')),         2e-4),
  ("r16",      dict(method="lora", rank=16, alpha=32, targets=('query', 'value')),         2e-4),
  ("r32",      dict(method="lora", rank=32, alpha=64, targets=('query', 'value')),         2e-4),
  ("alpha64",  dict(method="lora", rank=8,  alpha=64, targets=('query', 'value')),         2e-4),
]

# 관심 비교쌍 (A, B): A가 B보다 유의하게 나은가.
PAIRS = [("r32", "full"), ("r16", "r8"), ("alpha64", "r32"), ("alpha64", "full")]


def train_and_predict(name, kwargs, lr, train_dl, dev_dl, device, max_steps, seed=11711):
  seed_everything(seed)
  model = PEFTParaphrase(**kwargs).to(device)
  optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
  model.train()
  step = 0
  done = False
  while not done:
    for batch in train_dl:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      loss = F.cross_entropy(model(b_ids, b_mask), labels)
      loss.backward()
      optimizer.step()
      step += 1
      if step >= max_steps:
        done = True
        break
  model.eval()
  acc, _, y_pred, y_true, _ = model_eval_paraphrase(dev_dl, model, device)
  correct = (np.array(y_pred) == np.array(y_true)).astype(np.int8)
  print(f"  [{name:<8}] dev acc = {acc:.4f}", flush=True)
  del model, optimizer
  torch.cuda.empty_cache()
  return correct


def mcnemar(corr_a, corr_b):
  n10 = int(np.sum((corr_a == 1) & (corr_b == 0)))   # A만 맞음
  n01 = int(np.sum((corr_a == 0) & (corr_b == 1)))   # B만 맞음
  n = n10 + n01
  if n == 0:
    return n10, n01, float('nan'), 1.0
  stat = (abs(n10 - n01) - 1) ** 2 / n
  if n < 25:  # 불일치가 적으면 정확 이항검정.
    p = binomtest(min(n10, n01), n, 0.5).pvalue
  else:
    p = chi2.sf(stat, df=1)
  return n10, n01, stat, p


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=3000)
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
  print(f"dev {len(dev_ds):,} examples | seed 11711 | {args.max_steps} steps (예제 단위 McNemar)\n")

  print("=== 핵심 설정 학습·예측 (seed 11711) ===")
  corr = {}
  for name, kwargs, lr in CONFIGS:
    corr[name] = train_and_predict(name, kwargs, lr, train_dl, dev_dl, device, args.max_steps)

  print(f"\n========== 예제 단위 McNemar 검정 (dev n={len(dev_ds):,}) ==========")
  print(f"{'비교 (A vs B)':<22} {'A만맞음(n10)':>12} {'B만맞음(n01)':>12} {'χ²':>9} {'p-value':>12} {'유의(α=.05)':>12}")
  for a_name, b_name in PAIRS:
    n10, n01, stat, p = mcnemar(corr[a_name], corr[b_name])
    sig = "유의" if p < 0.05 else "유의X(노이즈)"
    direction = f"{a_name}↑" if n10 > n01 else f"{b_name}↑"
    print(f"{a_name+' vs '+b_name:<22} {n10:>12,} {n01:>12,} {stat:>9.2f} {p:>12.2e} {sig+' '+direction:>12}")


if __name__ == "__main__":
  main()
