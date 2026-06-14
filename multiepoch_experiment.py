"""
다중 에폭 수렴 비교 — "PEFT가 충분히 학습하면 full FT를 따라잡나" (Tier 2 #4).

§4.5의 동일-step(3,000≈0.17 epoch) 비교는 과소적합 영역이었다. 여기서는 full FT /
LoRA r=32 / LoRA α=64(r8)를 **1 epoch 완주**(batch 16, bf16)·3개 시드로 학습해
수렴 정확도를 비교한다. §4.4(단일 시드 1-epoch)를 다중 시드로 승격한다.

메모리 안전: 한 프로세스에서 여러 모델을 연속 생성하면 GPU 메모리가 완전히 해제되지
않아(누수) 8GB에서 spill이 난다(§4.7). 따라서 **run마다 별도 자식 프로세스**로 격리해
각 run 종료 시 OS가 GPU 메모리를 완전히 회수하도록 한다. bf16 autocast는 §4.7.1에서
정확도 무손실·가속을 검증했다.

추가로, 가변 길이 배치(Quora 문장쌍) 탓에 한 run **안에서도** 캐싱 할당자의 예약이
계속 자라(실측 6.2~6.4GB > §4.7.1의 peak 3.5GB) 다른 프로세스 점유와 합쳐 spill이
재발했다(100% util·~41W). expandable_segments는 Windows에서 효과가 없어,
**per-process 메모리 상한(72%) + garbage_collection_threshold + max_split_size_mb +
주기적 empty_cache**로 예약을 강제 억제한다(할당자 설정이라 학습 수치에는 영향 없음).
또한 자식 출력은 실시간 스트리밍해 진행 속도(it/s)를 로그에서 바로 볼 수 있게 한다.

윤리: train/dev만 사용, 공식 test 미사용.

실행:
  python multiepoch_experiment.py --use_gpu --batch_size 16 --epochs 1 --seeds 11711 42 1234
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

CONFIGS = {
  "full":    (dict(method="full"),                                                 1e-5),
  "r32":     (dict(method="lora", rank=32, alpha=64, targets=('query', 'value')),  2e-4),
  "alpha64": (dict(method="lora", rank=8,  alpha=64, targets=('query', 'value')),  2e-4),
}
ORDER = ["full", "r32", "alpha64"]


def worker(name, seed, args):
  """자식 프로세스: 한 (config, seed)만 학습·평가하고 RESULT 한 줄을 출력."""
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
    # 다른 프로세스 점유(~1.2GB)와 합쳐 8GB를 넘지 않도록 하드 캡 → 드라이버 RAM spill 차단.
    # full FT(batch 16)의 라이브 피크가 ~5.4GB+라 0.72(5.76GB)에선 OOM → 0.80(6.55GB).
    torch.cuda.set_per_process_memory_fraction(0.80)
  kwargs, lr = CONFIGS[name]

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
  total = len(train_dl) * args.epochs
  for _ in range(args.epochs):
    for batch in train_dl:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
          loss = F.cross_entropy(model(b_ids, b_mask), labels)
        loss.backward()
      except torch.cuda.OutOfMemoryError:
        # 캡 경계의 초장문 배치 1회 한정 안전장치: 캐시 비우고 같은 배치를 재시도.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(f"    [{name} s{seed}] step {step+1} OOM → empty_cache 후 재시도", flush=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
          loss = F.cross_entropy(model(b_ids, b_mask), labels)
        loss.backward()
      optimizer.step()
      step += 1
      if device.type == 'cuda' and step % 500 == 0:
        torch.cuda.empty_cache()  # 가변 길이 배치로 인한 예약 증식 → spill 방지
        print(f"    [{name} s{seed}] step {step}/{total} | {step/(time.time()-t0):.1f} it/s "
              f"| reserved {torch.cuda.memory_reserved()/2**30:.2f} GB", flush=True)
      if args.max_steps is not None and step >= args.max_steps:
        break
    else:
      continue
    break
  dt = time.time() - t0
  model.eval()
  acc, *_ = model_eval_paraphrase(dev_dl, model, device)
  print(f"  [{name:<8} s{seed}] {step} steps ({dt:5.0f}s, {step/dt:4.1f} it/s) | dev acc = {acc:.4f}", flush=True)
  print(f"RESULT\t{name}\t{seed}\t{acc:.6f}", flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--max_steps", type=int, default=None, help="smoke/안전용 step 상한")
  parser.add_argument("--seeds", type=int, nargs='+', default=[11711, 42, 1234])
  # 내부용: 자식 프로세스가 단일 run을 수행할 때 사용.
  parser.add_argument("--run_one", type=str, default=None, choices=list(CONFIGS))
  parser.add_argument("--seed_one", type=int, default=None)
  args = parser.parse_args()

  if args.run_one is not None:
    worker(args.run_one, args.seed_one, args)
    return

  # 오케스트레이터: run마다 자식 프로세스를 띄워 메모리를 완전히 격리한다.
  print(f"bf16 batch {args.batch_size} | {args.epochs} epoch × {len(ORDER)} configs × {len(args.seeds)} seeds "
        f"(run별 프로세스 격리)\n")
  accs = {name: [] for name in ORDER}
  for seed in args.seeds:
    print(f"\n########## seed = {seed} ##########", flush=True)
    for name in ORDER:
      cmd = [sys.executable, __file__, "--run_one", name, "--seed_one", str(seed),
             "--batch_size", str(args.batch_size), "--epochs", str(args.epochs),
             "--para_train", args.para_train, "--para_dev", args.para_dev]
      if args.use_gpu:
        cmd.append("--use_gpu")
      if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
      # 자식 출력을 실시간 스트리밍(stderr 포함)해 진행 속도를 로그에서 바로 확인한다.
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
        print(f"  [{name} s{seed}] 실패! (exit {proc.returncode})", flush=True)
        continue
      _, n, s, acc = result_line.split("\t")
      accs[name].append(float(acc))
      print(f"  [{name:<8} s{seed}] dev acc = {float(acc):.4f}", flush=True)

  print(f"\n\n========== {args.epochs}-epoch 수렴 비교 (dev acc, mean ± std) ==========")
  full = np.array(accs["full"]) if accs["full"] else np.array([np.nan])
  for name in ORDER:
    a = np.array(accs[name]) if accs[name] else np.array([np.nan])
    seedstr = " ".join(f"{x:.4f}" for x in a)
    delta = "" if name == "full" else f" | full 대비 Δ={a.mean()-full.mean():+.4f}"
    print(f"  {name:<8} mean {a.mean():.4f} ± {a.std(ddof=0):.4f}  (min {a.min():.4f}, max {a.max():.4f})  [{seedstr}]{delta}")


if __name__ == "__main__":
  main()
