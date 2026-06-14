"""
혼합정밀(mixed precision) 실험 — §4.7 메모리 제약의 후속 검증.

§4.7에서 full fine-tuning을 batch 16으로 돌리자 GPT-2 124M의 파라미터+그래디언트+
AdamW 상태+활성값이 8GB VRAM을 초과해 Windows가 시스템 RAM으로 spill → 약 20배
저하되는 현상을 관측했다. 보고서 §5.3은 이를 "mixed precision(fp16/bf16)으로 8GB에서
full FT batch 16 적재 가능 여부 검토"라는 향후 과제로 남겼다. 본 드라이버가 이를 실측한다.

full fine-tuning을 {fp32, bf16} × {batch 8, batch 16} 로 짧게 학습하며
  - it/s (처리 속도)
  - torch.cuda.max_memory_allocated (peak VRAM)
  - dev accuracy (bf16이 정확도를 해치지 않는지)
를 측정·비교한다. bf16 autocast는 파라미터를 fp32로 유지하고 forward만 bf16으로
계산하므로 직접 구현한 AdamW와 그대로 호환되며 GradScaler가 불필요하다.

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python amp_experiment.py --use_gpu --steps 300
"""

import argparse
import sys
import time

try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import evaluation
evaluation.TQDM_DISABLE = True

from datasets import ParaphraseDetectionDataset, load_paraphrase_data
from evaluation import model_eval_paraphrase
from optimizer import AdamW
from lora_experiments import PEFTParaphrase, seed_everything


def run(precision, batch_size, train_raw, dev_raw, args, device, steps):
  """full FT 한 설정을 학습하며 it/s·peak VRAM·dev acc 측정."""
  seed_everything(11711)
  # batch_size 가 데이터셋 토큰화에 쓰이므로 args에 반영해 DataLoader 구성.
  args.batch_size = batch_size
  train_ds = ParaphraseDetectionDataset(train_raw, args)
  dev_ds = ParaphraseDetectionDataset(dev_raw, args)
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=train_ds.collate_fn)
  dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, collate_fn=dev_ds.collate_fn)

  use_bf16 = (precision == 'bf16')
  model = PEFTParaphrase('full').to(device)
  optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)

  torch.cuda.reset_peak_memory_stats(device)
  model.train()
  t0 = time.time()
  step = 0
  done = False
  while not done:
    for batch in train_dl:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      step += 1
      if step >= steps:
        done = True
        break
  torch.cuda.synchronize(device)
  dt = time.time() - t0
  its = step / dt
  alloc_gb = torch.cuda.max_memory_allocated(device) / 1e9
  reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9

  model.eval()
  dev_acc, *_ = model_eval_paraphrase(dev_dl, model, device)

  print(f"  [{precision:>4} / batch {batch_size:>2}] {step} steps @ {its:5.2f} it/s ({dt:5.0f}s) | "
        f"peak alloc {alloc_gb:4.2f} GB | peak reserved {reserved_gb:4.2f} GB | dev acc = {dev_acc:.4f}",
        flush=True)
  del model, optimizer
  torch.cuda.empty_cache()
  return {'precision': precision, 'batch': batch_size, 'its': its,
          'alloc_gb': alloc_gb, 'reserved_gb': reserved_gb, 'dev_acc': dev_acc}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--steps", type=int, default=300)
  args = parser.parse_args()

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  seed_everything(11711)

  print("Quora train/dev 로드 중...")
  train_raw = load_paraphrase_data(args.para_train)
  dev_raw = load_paraphrase_data(args.para_dev)
  print(f"VRAM 총량 {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB | "
        f"full FT × {args.steps} steps\n")

  print("=== full fine-tuning: precision × batch_size 메모리/속도 ===")
  results = []
  for precision in ('fp32', 'bf16'):
    for batch in (8, 16):
      results.append(run(precision, batch, train_raw, dev_raw, args, device, args.steps))

  print("\n========== 요약 ==========")
  print(f"{'precision':<10} {'batch':>6} {'it/s':>7} {'alloc(GB)':>10} {'reserved(GB)':>13} {'dev acc':>9}")
  for r in results:
    print(f"{r['precision']:<10} {r['batch']:>6} {r['its']:>7.2f} {r['alloc_gb']:>10.2f} "
          f"{r['reserved_gb']:>13.2f} {r['dev_acc']:>9.4f}")


if __name__ == "__main__":
  main()
