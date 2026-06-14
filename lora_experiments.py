"""
PART-II 확장 실험 드라이버 (Paraphrase / PEFT).

한 번 로드한 Quora train/dev로 여러 PEFT 설정을 연속 학습·평가한다.
공정 비교를 위해 ablation 설정들은 동일한 학습 step 수(--max_steps)로 학습한다.

실험:
  (A) LoRA 주입 대상 모듈 ablation : query/value → +key → +attn_out → +mlp
  (B) LoRA rank ablation          : r ∈ {4, 8, 16, 32}
  (C) ReFT (LoReFT) 비교           : rank ∈ {4, 8}
  (D) 최적 설정 다중 에폭 풀 학습     : --full_epochs 에폭 (전체 데이터)

윤리: train/dev만 사용. 공식 test 미사용.

실행:
  python lora_experiments.py --use_gpu                 # 전체 (A~D)
  python lora_experiments.py --use_gpu --only ablation # A~C(step-matched)만
"""

import argparse
import random
import sys
import time

# Windows 콘솔(cp949)에서 한글/em-dash 출력 시 크래시 방지.
try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW
from lora import inject_lora
from reft import inject_reft


def seed_everything(seed=11711):
  random.seed(seed); np.random.seed(seed)
  torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class PEFTParaphrase(nn.Module):
  """Cloze 방식 Paraphrase 모델 + 선택적 PEFT(full / lora / reft)."""

  def __init__(self, method, rank=8, alpha=16, targets=('query', 'value'),
               dropout=0.0, reft_rank=4):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model='gpt2', d=768, l=12, num_heads=12)

    if method == 'full':
      for p in self.gpt.parameters():
        p.requires_grad = True
      self.info = "full fine-tuning"
    elif method == 'lora':
      for p in self.gpt.parameters():
        p.requires_grad = False
      n, tr, tot = inject_lora(self.gpt, rank=rank, alpha=alpha, targets=targets, dropout=dropout)
      self.info = f"LoRA r={rank} a={alpha} targets={'+'.join(targets)} drop={dropout}"
    elif method == 'reft':
      n, tr, tot = inject_reft(self.gpt, rank=reft_rank)
      self.info = f"ReFT(LoReFT) rank={reft_rank} (all layers)"
    else:
      raise ValueError(method)

  def forward(self, input_ids, attention_mask):
    last_token = self.gpt(input_ids, attention_mask)['last_token']
    return self.gpt.hidden_state_to_token(last_token)


def param_stats(model):
  tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
  tot = sum(p.numel() for p in model.parameters())
  return tr, tot


def run_config(name, method, train_dl, dev_dl, device, lr,
               rank=8, alpha=16, targets=('query', 'value'), dropout=0.0,
               reft_rank=4, max_steps=None, epochs=1, seed=11711):
  """한 설정을 학습·평가하고 결과 dict 반환."""
  seed_everything(seed)
  model = PEFTParaphrase(method, rank, alpha, targets, dropout, reft_rank).to(device)
  tr, tot = param_stats(model)
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(params, lr=lr, weight_decay=0.)

  print(f"  [{name}] 학습 시작 (trainable {tr:,}, {100*tr/tot:.3f}%)", flush=True)
  model.train()
  t0 = time.time()
  step = 0
  done = False
  for epoch in range(epochs):
    if done:
      break
    for batch in train_dl:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      step += 1
      if step % 500 == 0:
        el = time.time() - t0
        print(f"    {name} step {step} | loss {loss.item():.3f} | {step/el:.1f} it/s", flush=True)
      if max_steps is not None and step >= max_steps:
        done = True
        break
  train_time = time.time() - t0
  its = step / train_time if train_time > 0 else 0.0

  model.eval()
  dev_acc, *_ = model_eval_paraphrase(dev_dl, model, device)

  pct = 100.0 * tr / tot
  print(f"  [{name:<32}] {model.info:<46} | 학습파라미터 {tr:>11,} ({pct:6.3f}%) | "
        f"{step} steps @ {its:4.1f} it/s ({train_time:5.0f}s) | dev acc = {dev_acc:.4f}")

  del model, optimizer
  torch.cuda.empty_cache()
  return {'name': name, 'info': '', 'trainable': tr, 'pct': pct,
          'steps': step, 'its': its, 'time': train_time, 'dev_acc': dev_acc}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--max_steps", type=int, default=3000,
                      help="ablation(A~C) 설정의 step-matched 학습 step 수")
  parser.add_argument("--full_steps", type=int, default=8000,
                      help="(D) 최적 설정 확장 학습 step 수 (~0.45 epoch)")
  parser.add_argument("--only", type=str, default="all", choices=["all", "ablation", "full"])
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
  print(f"train {len(train_ds):,} / dev {len(dev_ds):,} examples 로드 완료\n")

  results = []
  ms = args.max_steps

  if args.only in ("all", "ablation"):
    print(f"=== (A) 모듈 ablation — LoRA r=8 a=16 lr=2e-4, {ms} steps 동일 ===")
    results.append(run_config("A:full(ref)", "full", train_dl, dev_dl, device, lr=1e-5, max_steps=ms))
    results.append(run_config("A:lora qv", "lora", train_dl, dev_dl, device, lr=2e-4,
                              rank=8, alpha=16, targets=('query', 'value'), max_steps=ms))
    results.append(run_config("A:lora qkv", "lora", train_dl, dev_dl, device, lr=2e-4,
                              rank=8, alpha=16, targets=('query', 'key', 'value'), max_steps=ms))
    results.append(run_config("A:lora qkv+out", "lora", train_dl, dev_dl, device, lr=2e-4,
                              rank=8, alpha=16, targets=('query', 'key', 'value', 'attn_out'), max_steps=ms))
    results.append(run_config("A:lora all", "lora", train_dl, dev_dl, device, lr=2e-4,
                              rank=8, alpha=16, targets=('query', 'key', 'value', 'attn_out', 'mlp'), max_steps=ms))

    print(f"\n=== (B) rank ablation — LoRA targets=qv lr=2e-4, {ms} steps 동일 ===")
    for r in (4, 16, 32):
      results.append(run_config(f"B:lora r={r}", "lora", train_dl, dev_dl, device, lr=2e-4,
                                rank=r, alpha=2 * r, targets=('query', 'value'), max_steps=ms))

    print(f"\n=== (C) ReFT 비교 — {ms} steps 동일 ===")
    for rr in (4, 8):
      results.append(run_config(f"C:reft r={rr}", "reft", train_dl, dev_dl, device, lr=2e-4,
                                reft_rank=rr, max_steps=ms))

  if args.only in ("all", "full"):
    print(f"\n=== (D) 최적 설정 확장 학습 — LoRA all-modules r=16 a=32, {args.full_steps} steps ===")
    results.append(run_config(f"D:lora all r16 {args.full_steps}st", "lora", train_dl, dev_dl, device,
                              lr=3e-4, rank=16, alpha=32,
                              targets=('query', 'key', 'value', 'attn_out', 'mlp'),
                              max_steps=args.full_steps, epochs=10))

  print("\n\n========== 결과 요약 (dev accuracy 내림차순) ==========")
  results.sort(key=lambda d: -d['dev_acc'])
  print(f"{'설정':<28} {'학습파라미터':>13} {'비율%':>8} {'steps':>7} {'it/s':>6} {'dev acc':>9}")
  for d in results:
    print(f"{d['name']:<28} {d['trainable']:>13,} {d['pct']:>8.3f} {d['steps']:>7} {d['its']:>6.1f} {d['dev_acc']:>9.4f}")


if __name__ == "__main__":
  main()
