"""
Sonnet 디코딩 — 다중 시드 chrF + beam search 비교 (PART-II 확장 심화).

기존 sonnet_decode_experiment.py 는 단일 시드(11711)로 디코딩 전략을 sweep했다.
확률적 샘플링은 시드에 따라 chrF가 흔들리므로, 본 드라이버는:
  (1) 대표 확률적 설정을 여러 시드로 반복해 chrF 평균±표준편차를 보고
      → §4.6 순위(특히 greedy→샘플링 +10.9, repetition penalty 음성 결과)가
        시드에 견고한지 검증한다.
  (2) 결정론적 beam search(beam ∈ {1,3,5})를 추가해, 탐욕적 탐색 강화가
      chrF에서 확률적 샘플링을 이기는지 비교한다.

윤리: 학습 소네트의 마지막 DEV_SIZE편을 자체 held-out으로만 사용, 공식 test 미사용.

실행:
  python sonnet_decode_multiseed.py --use_gpu --seeds 11711 42 1234
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
from sacrebleu.metrics import CHRF

from datasets import SonnetsDataset
from sonnet_generation import SonnetGPT


def seed_everything(seed):
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def first_n_lines(sonnet_text, n=3):
  return '\n'.join(sonnet_text.split('\n')[:n])


@torch.no_grad()
def eval_sampling(model, dev, device, temperature, top_p, top_k, rep_penalty, seed, max_length=128):
  """확률적 디코딩 한 설정을 주어진 시드로 dev 전체 생성 → corpus chrF."""
  chrf = CHRF()
  hyps, refs = [], []
  for true_sonnet in dev:
    seed_everything(seed)  # 시드 내에서는 소네트 간 동일 시드(설정 간 공정 비교 유지).
    prompt = first_n_lines(true_sonnet, 3)
    enc = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    out_ids = model.generate(enc['input_ids'], temperature=temperature, top_p=top_p,
                             top_k=top_k, max_length=max_length, repetition_penalty=rep_penalty)[0][0]
    hyps.append(model.tokenizer.decode(out_ids))
    refs.append(true_sonnet)
  return chrf.corpus_score(hyps, [refs]).score


@torch.no_grad()
def eval_beam(model, dev, device, beam_size, length_penalty=1.0, max_length=128):
  """결정론적 beam search → corpus chrF (시드 무관)."""
  chrf = CHRF()
  hyps, refs = [], []
  for true_sonnet in dev:
    prompt = first_n_lines(true_sonnet, 3)
    enc = model.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(device)
    out_ids = model.generate_beam(enc['input_ids'], beam_size=beam_size,
                                  max_length=max_length, length_penalty=length_penalty)
    hyps.append(model.tokenizer.decode(out_ids))
    refs.append(true_sonnet)
  return chrf.corpus_score(hyps, [refs]).score


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt", type=str, default="9_10-1e-05-sonnet.pt")
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--dev_size", type=int, default=14)
  parser.add_argument("--seeds", type=int, nargs='+', default=[11711, 42, 1234])
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
  print(f"Dev held-out 소네트 {len(dev)}편 | seeds={args.seeds}\n")

  # (name, temperature, top_p, top_k, rep_penalty)
  sampling_cfgs = [
    ("best: temp=1.2 top-p=0.9", 1.2, 0.9, 0, 1.0),
    ("temp=1.0",                 1.0, 1.0, 0, 1.0),
    ("top-p=0.9 temp=0.9",       0.9, 0.9, 0, 1.0),
    ("best + rep_penalty=1.2",   1.2, 0.9, 0, 1.2),
  ]

  print("=== (1) 확률적 디코딩 — 다중 시드 chrF ===")
  rows = []
  for name, t, p, k, rep in sampling_cfgs:
    vals = np.array([eval_sampling(model, dev, device, t, p, k, rep, s) for s in args.seeds])
    rows.append((name, vals.mean(), vals.std(ddof=0), vals.min(), vals.max(), vals))
    seedstr = " ".join(f"{x:.2f}" for x in vals)
    print(f"  {name:<26} chrF = {vals.mean():.2f} ± {vals.std(ddof=0):.2f}  "
          f"(min {vals.min():.2f}, max {vals.max():.2f})  시드별 [{seedstr}]")

  print("\n=== (2) beam search — 결정론적 chrF ===")
  beam_rows = []
  for b in (1, 3, 5):
    score = eval_beam(model, dev, device, beam_size=b)
    label = f"beam={b}" + (" (=greedy)" if b == 1 else "")
    beam_rows.append((label, score))
    print(f"  {label:<26} chrF = {score:.2f}")

  print("\n=== 종합 (chrF 내림차순) ===")
  combined = [(n, m, s) for (n, m, s, _, _, _) in rows] + [(n, sc, 0.0) for (n, sc) in beam_rows]
  combined.sort(key=lambda x: -x[1])
  for rank, (n, m, s) in enumerate(combined, 1):
    pm = f" ± {s:.2f}" if s > 0 else "      (결정론적)"
    print(f"  {rank:>2}. {n:<26} chrF = {m:.2f}{pm}")


if __name__ == "__main__":
  main()
