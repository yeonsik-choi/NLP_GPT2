"""
확장 학습 D(all-modules r16, 8,000 step)의 다중 시드 승격 — §5.2 한계 마감.

§4.5의 D 설정(LoRA r=16 α=32, q/k/v+attn_out+mlp, **lr 3e-4**, batch 8, 8,000 step,
fp32 — `lora_experiments.py`의 D 호출과 동일 프로토콜. 주의: ablation A~C의 lr 2e-4와
달리 D만 3e-4였다)을 시드 {11711, 42, 1234}로 반복한다.
seed 11711은 §4.5의 0.8186을 재현해야 한다(드라이버 검증용 앵커).
(부수 관찰: lr 2e-4로 잘못 돌린 첫 시도에서 s11711=0.8296이 나왔다 — d8000_lr2e4.log)

메모리 안전(§4.5.6 실험 노트와 동일): run별 자식 프로세스 격리 + per-process 상한 +
garbage_collection_threshold + 주기적 empty_cache. batch 8 LoRA라 여유가 크지만
동일 레시피를 유지한다.

실행:
  python d8000_multiseed.py --use_gpu --batch_size 8 --max_steps 8000 --seeds 11711 42 1234
"""

import argparse
import os
import subprocess
import sys
import time

# torch가 CUDA를 초기화하기 전에 설정해야 효과가 있다(자식도 이 파일을 재실행하므로 적용됨).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:64'

try:
  sys.stdout.reconfigure(encoding='utf-8')
except Exception:
  pass

import numpy as np

# §4.5 D와 동일: all-modules r16 α32, lr 3e-4 (fp32) — lora_experiments.py L189 참조.
CONFIG = (dict(method="lora", rank=16, alpha=32,
               targets=('query', 'key', 'value', 'attn_out', 'mlp')), 3e-4)


def worker(seed, args):
  """자식 프로세스: 한 시드만 학습·평가하고 RESULT 한 줄을 출력."""
  import torch
  import torch.nn.functional as F
  from torch.utils.data import DataLoader
  import evaluation
  evaluation.TQDM_DISABLE = True
  from datasets import ParaphraseDetectionDataset, load_paraphrase_data
  from evaluation import model_eval_paraphrase
  from optimizer import AdamW
  from lora_experiments import PEFTParaphrase, seed_everything

  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if device.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.80)
  kwargs, lr = CONFIG

  train_raw = load_paraphrase_data(args.para_train)
  dev_raw = load_paraphrase_data(args.para_dev)
  train_ds = ParaphraseDetectionDataset(train_raw, args)
  dev_ds = ParaphraseDetectionDataset(dev_raw, args)
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=train_ds.collate_fn)
  dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=args.batch_size, collate_fn=dev_ds.collate_fn)

  seed_everything(seed)
  model = PEFTParaphrase(**kwargs).to(device)
  optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
  model.train()
  t0 = time.time()
  step = 0
  done = False
  while not done:  # 8,000 step > 1 epoch(batch 8 기준 ~35K step)은 아니므로 1 epoch 내에서 끝난다.
    for batch in train_dl:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      loss = F.cross_entropy(model(b_ids, b_mask), labels)
      loss.backward()
      optimizer.step()
      step += 1
      if device.type == 'cuda' and step % 500 == 0:
        torch.cuda.empty_cache()
        print(f"    [d8000 s{seed}] step {step}/{args.max_steps} | {step/(time.time()-t0):.1f} it/s "
              f"| reserved {torch.cuda.memory_reserved()/2**30:.2f} GB", flush=True)
      if step >= args.max_steps:
        done = True
        break
  dt = time.time() - t0
  model.eval()
  acc, *_ = model_eval_paraphrase(dev_dl, model, device)
  print(f"  [d8000 s{seed}] {step} steps ({dt:5.0f}s, {step/dt:4.1f} it/s) | dev acc = {acc:.4f}", flush=True)
  print(f"RESULT\td8000\t{seed}\t{acc:.6f}", flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=8000)
  parser.add_argument("--seeds", type=int, nargs='+', default=[11711, 42, 1234])
  parser.add_argument("--seed_one", type=int, default=None)  # 내부용: 자식 프로세스 단일 run.
  args = parser.parse_args()

  if args.seed_one is not None:
    worker(args.seed_one, args)
    return

  print(f"D 확장 학습(all-modules r16 a32, fp32) batch {args.batch_size} | {args.max_steps} steps "
        f"× {len(args.seeds)} seeds (run별 프로세스 격리)\n")
  accs = []
  for seed in args.seeds:
    cmd = [sys.executable, __file__, "--seed_one", str(seed),
           "--batch_size", str(args.batch_size), "--max_steps", str(args.max_steps),
           "--para_train", args.para_train, "--para_dev", args.para_dev]
    if args.use_gpu:
      cmd.append("--use_gpu")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
    result_line = None
    for l in proc.stdout:
      l = l.rstrip()
      if l.startswith("RESULT\t"):
        result_line = l
      elif l:
        print(l, flush=True)
    proc.wait()
    if result_line is None:
      print(f"  [d8000 s{seed}] 실패! (exit {proc.returncode})", flush=True)
      continue
    accs.append(float(result_line.split("\t")[3]))
    print(f"  [d8000 s{seed}] dev acc = {accs[-1]:.4f}", flush=True)

  a = np.array(accs) if accs else np.array([np.nan])
  print(f"\n========== D(8,000 step) 다중 시드 (dev acc) ==========")
  print(f"  mean {a.mean():.4f} ± {a.std(ddof=0):.4f}  (min {a.min():.4f}, max {a.max():.4f})  "
        f"[{' '.join(f'{x:.4f}' for x in a)}]")


if __name__ == "__main__":
  main()
