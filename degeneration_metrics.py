"""
생성 퇴화(degeneration) 정량화 — beam이 chrF에서 지는 '이유'를 증거화 (Tier 1 #2).

§4.6.1은 beam search가 샘플링에 ~10점 뒤지는 것을 "반복적·저다양성" 때문이라
주장했으나 수치로 보이진 않았다. 여기서 각 디코딩 전략의 생성문에서 다양성·반복
지표를 직접 측정해 chrF 격차의 메커니즘을 입증한다 [Holtzman et al., 2020].

지표(생성문 토큰 기준, dev 14편 평균):
  - distinct-1 / distinct-2 : 고유 unigram/bigram 비율(↑ 다양)
  - rep-4                   : 반복되는 4-gram 비율(↑ 퇴화)
  - mean_len               : 생성 토큰 길이

윤리: 학습 소네트 마지막 14편(자체 held-out)만 사용, 공식 test 미사용.

실행:
  python degeneration_metrics.py --use_gpu
"""

import argparse
import random
import sys

try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import numpy as np
import torch
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT


def seed_everything(seed=11711):
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def first_n_lines(t, n=3):
  return '\n'.join(t.split('\n')[:n])


def distinct_n(tokens, n):
  if len(tokens) < n:
    return 0.0
  ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
  return len(set(ngrams)) / len(ngrams)


def rep_n(tokens, n=4):
  """반복되는 n-gram 비율 = 1 - distinct-n (중복이 많을수록 ↑)."""
  if len(tokens) < n:
    return 0.0
  return 1.0 - distinct_n(tokens, n)


def metrics_over(token_lists):
  d1 = np.mean([distinct_n(t, 1) for t in token_lists])
  d2 = np.mean([distinct_n(t, 2) for t in token_lists])
  r4 = np.mean([rep_n(t, 4) for t in token_lists])
  ml = np.mean([len(t) for t in token_lists])
  return d1, d2, r4, ml


@torch.no_grad()
def gen_sampling(model, dev, device, temperature, top_p, rep_penalty):
  outs = []
  for true_sonnet in dev:
    seed_everything(11711)
    enc = model.tokenizer(first_n_lines(true_sonnet, 3), return_tensors='pt', truncation=True).to(device)
    ids, _ = model.generate(enc['input_ids'], temperature=temperature, top_p=top_p,
                            top_k=0, max_length=128, repetition_penalty=rep_penalty)
    prompt_len = enc['input_ids'].shape[1]
    outs.append(ids[0][prompt_len:].cpu().tolist())   # 생성분만(프롬프트 제외).
  return outs


@torch.no_grad()
def gen_beam(model, dev, device, beam_size):
  outs = []
  for true_sonnet in dev:
    enc = model.tokenizer(first_n_lines(true_sonnet, 3), return_tensors='pt', truncation=True).to(device)
    ids = model.generate_beam(enc['input_ids'], beam_size=beam_size, max_length=128)
    prompt_len = enc['input_ids'].shape[1]
    outs.append(ids[prompt_len:].cpu().tolist())
  return outs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, default="9_10-1e-05-sonnet.pt")
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--dev_size", type=int, default=14)
  parser.add_argument("--use_gpu", action='store_true')
  args = parser.parse_args()

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.ckpt, weights_only=False)
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device).eval()
  print(f"Loaded checkpoint: {args.ckpt}")

  all_sonnets = [s for (_, s) in SonnetsDataset(args.sonnet_path)]
  dev = all_sonnets[-args.dev_size:]
  print(f"Dev held-out {len(dev)}편 | 생성문 다양성·반복 지표\n")

  rows = []
  rows.append(("best: temp1.2 top-p0.9 (chrF 42.1)", gen_sampling(model, dev, device, 1.2, 0.9, 1.0)))
  rows.append(("rep_penalty=1.2 (chrF 39.8)",        gen_sampling(model, dev, device, 1.2, 0.9, 1.2)))
  rows.append(("beam=3 (chrF 32.2)",                 gen_beam(model, dev, device, 3)))
  rows.append(("greedy (chrF 31.7)",                 gen_beam(model, dev, device, 1)))

  print(f"{'전략':<36} {'distinct-1':>11} {'distinct-2':>11} {'rep-4':>8} {'mean_len':>9}")
  for name, outs in rows:
    d1, d2, r4, ml = metrics_over(outs)
    print(f"{name:<36} {d1:>11.3f} {d2:>11.3f} {r4:>8.3f} {ml:>9.1f}")


if __name__ == "__main__":
  main()
