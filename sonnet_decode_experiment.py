"""
PART-II 확장 (Sonnet Generation): 디코딩 전략의 정량 비교.

소네트 평가지표는 chrF이다. 공식 test 소네트(grader 보유)는 사용할 수 없으므로,
학습 소네트의 일부(마지막 DEV_SIZE편)를 자체 held-out으로 떼어내 dev chrF를 측정한다.
각 dev 소네트의 처음 3줄을 조건으로 나머지를 생성하고, 생성문 전체를 원본 소네트와 chrF로 비교한다.
(이 과정은 train 데이터만 사용 → 윤리 강령 준수)

실행:
  python sonnet_decode_experiment.py --use_gpu
"""

import argparse
import random

import numpy as np
import torch
from sacrebleu.metrics import CHRF

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT


def seed_everything(seed=11711):
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def first_n_lines(sonnet_text, n=3):
  lines = sonnet_text.split('\n')
  return '\n'.join(lines[:n])


@torch.no_grad()
def eval_config(model, dev, device, name, temperature, top_p, top_k, rep_penalty=1.0, max_length=128):
  chrf = CHRF()
  hyps, refs = [], []
  for true_sonnet in dev:
    seed_everything(11711)  # 설정 간 공정 비교를 위해 동일 시드.
    prompt = first_n_lines(true_sonnet, 3)
    enc = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    out_ids = model.generate(enc['input_ids'], temperature=temperature, top_p=top_p,
                             top_k=top_k, max_length=max_length,
                             repetition_penalty=rep_penalty)[0][0]
    hyps.append(model.tokenizer.decode(out_ids))
    refs.append(true_sonnet)
  score = chrf.corpus_score(hyps, [refs]).score
  print(f"  {name:<38} temp={temperature:<4} top_p={top_p:<4} top_k={top_k:<4} rep={rep_penalty:<4} | chrF = {score:.2f}")
  return name, score, hyps[0]


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
  print(f"Dev held-out 소네트 {len(dev)}편으로 chrF 측정 (train 데이터에서 분리)\n")

  # 디코딩 전략 sweep. (name, temperature, top_p, top_k, repetition_penalty)
  configs = [
    ("greedy",                       1.0, 1.0, 1,  1.0),
    ("temperature=0.7",              0.7, 1.0, 0,  1.0),
    ("temperature=1.0",              1.0, 1.0, 0,  1.0),
    ("temperature=1.2 (기본값)",      1.2, 0.9, 0,  1.0),
    ("top-p=0.9, temp=0.9",          0.9, 0.9, 0,  1.0),
    ("top-p=0.9, temp=0.7",          0.7, 0.9, 0,  1.0),
    ("top-k=40, temp=0.9",           0.9, 1.0, 40, 1.0),
    ("top-k=40 + top-p=0.9",         0.9, 0.9, 40, 1.0),
    # PART-II 확장: repetition penalty 추가.
    ("best + rep_penalty=1.2",       1.2, 0.9, 0,  1.2),
    ("best + rep_penalty=1.3",       1.2, 0.9, 0,  1.3),
    ("top-p0.9 t1.0 + rep=1.2",      1.0, 0.9, 0,  1.2),
  ]

  print("=== 디코딩 전략별 dev chrF ===")
  results = []
  best_sample = None
  for name, temp, top_p, top_k, rep in configs:
    n, score, sample = eval_config(model, dev, device, name, temp, top_p, top_k, rep_penalty=rep)
    results.append((n, score))
    if best_sample is None or score > best_sample[1]:
      best_sample = (n, score, sample)

  results.sort(key=lambda x: -x[1])
  print("\n=== 순위 (chrF 내림차순) ===")
  for rank, (n, score) in enumerate(results, 1):
    print(f"  {rank}. {n:<34} chrF = {score:.2f}")

  print(f"\n=== 최고 설정 '{best_sample[0]}' (chrF={best_sample[1]:.2f}) 생성 예시 ===\n")
  print(best_sample[2])


if __name__ == "__main__":
  main()
