'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    # 시퀀스의 각 토큰에 대한 contextualized embedding.
    hidden_states = self.gpt(input_ids, attention_mask)['last_hidden_state']
    # weight tying을 통해 각 위치에서 전체 vocab에 대한 logit을 계산. [bs, seq_len, vocab].
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=0, max_length=128,
               repetition_penalty=1.0):
    """
    temperature scaling과 함께 top-k / top-p(nucleus) 샘플링으로 새로운 소넷을 생성한다.

    PART-II 확장: 기존에는 top-p만 지원했으나, top_k 인자를 추가해 두 필터를 함께 적용할 수 있게 했다.
      - top_k=1   → greedy decoding (항상 argmax)
      - top_k>1   → 상위 k개 토큰으로 후보 제한
      - top_p<1.0 → 누적확률 p 이내의 nucleus로 후보 제한
    추가 확장: repetition_penalty(>1.0) 로 이미 생성된 토큰의 logit을 감쇠시켜 반복/퇴화를 억제한다
      [Keskar et al., 2019, CTRL]. chrF에 대한 효과를 sonnet_decode_experiment.py 에서 비교한다.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :].clone()

      # Repetition penalty: 이미 등장한 토큰의 logit을 감쇠 (양수는 나누고 음수는 곱한다).
      if repetition_penalty != 1.0:
        for prev in set(token_ids[0].tolist()):
          if logits_last_token[0, prev] > 0:
            logits_last_token[0, prev] /= repetition_penalty
          else:
            logits_last_token[0, prev] *= repetition_penalty

      logits_last_token = logits_last_token / temperature  # Apply temperature scaling

      # 내림차순 정렬 후 top-k / top-p 필터를 차례로 적용.
      sorted_logits, sorted_indices = torch.sort(logits_last_token, descending=True)
      sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

      keep = torch.ones_like(sorted_probs, dtype=torch.bool)
      # Top-k: 상위 k개만 유지.
      if top_k > 0:
        keep[..., top_k:] = False
      # Top-p (nucleus): 누적확률이 p를 넘는 토큰 제거(최상위 토큰은 항상 유지).
      if top_p < 1.0:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= top_p
        top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
        top_p_mask[..., 0] = True
        keep = keep & top_p_mask

      filtered_probs = sorted_probs * keep
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

  @torch.no_grad()
  def generate_beam(self, encoding, beam_size=5, max_length=128, length_penalty=1.0):
    """결정론적 beam search 디코딩 (확률적 샘플링과의 chrF 비교용, PART-II 확장).

    각 step마다 살아있는 beam을 vocab 전체로 확장해 누적 log-prob 상위 beam_size개만 유지한다.
    EOS에 도달한 beam은 고정하고, 길이 정규화(score / len^length_penalty)로 최종 beam을 고른다.
    beam_size=1 이면 greedy와 동일하다.
    """
    device = self.get_device()
    eos = self.tokenizer.eos_token_id
    seq = encoding.to(device)                                  # [1, L]
    beams = seq.repeat(beam_size, 1)                           # [B, L]
    scores = torch.full((beam_size,), -1e9, device=device)
    scores[0] = 0.0                                            # 시작 시 첫 beam만 활성(중복 방지).
    gen_len = torch.zeros(beam_size, device=device)
    done = torch.zeros(beam_size, dtype=torch.bool, device=device)

    for _ in range(max_length):
      mask = torch.ones_like(beams)
      logits = self.forward(beams, mask)[:, -1, :]            # [B, V]
      logp = torch.log_softmax(logits, dim=-1)
      vocab = logp.size(-1)
      # 이미 끝난 beam은 EOS만 점수 변화 없이 이어 붙인다.
      logp[done] = -1e9
      logp[done, eos] = 0.0
      live_before = ~done
      cand = (scores.unsqueeze(1) + logp).view(-1)            # [B*V]
      top_scores, top_idx = cand.topk(beam_size)
      beam_id = top_idx // vocab
      tok_id = top_idx % vocab
      beams = torch.cat([beams[beam_id], tok_id.unsqueeze(1)], dim=1)
      scores = top_scores
      gen_len = gen_len[beam_id] + live_before[beam_id].float()
      done = done[beam_id] | (tok_id == eos)
      if done.all():
        break

    final = scores / gen_len.clamp(min=1).pow(length_penalty)
    best = beams[final.argmax()]                              # [seq]
    return best


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
      labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+", encoding='utf-8') as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  generate_submission_sonnets(args)