"""
LoRA (Low-Rank Adaptation) — PART-II 확장.

참고: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).

사전학습된 가중치 W (frozen)에 저랭크 업데이트 ΔW = (alpha/r) * B @ A 만 학습한다.
- A: [r, in],  B: [out, r],  r << min(in, out)
- B를 0으로 초기화하므로 학습 시작 시 ΔW = 0 → 사전학습 모델과 동일한 출력에서 출발.

전체 fine-tuning(124M 파라미터 전부 갱신) 대신, 어텐션/MLP의 선형 프로젝션에만
LoRA를 주입하여 학습 파라미터를 대폭 줄인다.

업그레이드(PART-II 확장 ②):
  - 주입 대상 모듈을 선택 가능: query/key/value(어텐션 입력), attn_out(어텐션 출력 dense),
    mlp(피드포워드 두 dense). 기본값은 원조 LoRA 논문과 동일한 query/value.
  - LoRA dropout 지원.
"""

import math
import torch
from torch import nn


# 주입 가능한 모듈 이름. lora_experiments.py / paraphrase_detection.py 에서 참조.
ALL_TARGETS = ('query', 'key', 'value', 'attn_out', 'mlp')


class LoRALinear(nn.Module):
  """기존 nn.Linear를 감싸 frozen 가중치 + 학습 가능한 저랭크 보정항을 더한다."""

  def __init__(self, base_linear: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
    super().__init__()
    self.base = base_linear
    # 원본 가중치는 동결.
    for p in self.base.parameters():
      p.requires_grad = False

    # 주의: from_pretrained가 HF 가중치를 로드하면서 일부 층(MLP)의 .in_features/.out_features
    # 속성이 실제 weight와 어긋나 있다. 신뢰할 수 있는 weight shape [out, in]에서 직접 읽는다.
    out_features, in_features = base_linear.weight.shape
    self.rank = rank
    self.scaling = alpha / rank
    self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # 저랭크 행렬. A는 kaiming, B는 0 → 초기 ΔW = 0.
    self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
    self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

  def forward(self, x):
    base_out = self.base(x)
    lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
    return base_out + lora_out


def inject_lora(gpt_model, rank=8, alpha=16, targets=('query', 'value'), dropout=0.0):
  """GPT2Model의 각 트랜스포머 블록에서 선택된 선형층을 LoRALinear로 교체.

  targets 에 포함될 수 있는 이름:
    'query', 'key', 'value' → 어텐션 입력 프로젝션
    'attn_out'              → 어텐션 출력 dense (attention_dense)
    'mlp'                   → 피드포워드의 interm_dense + out_dense

  반환: (LoRA를 적용한 모듈 수, 학습 가능 파라미터 수, 전체 파라미터 수)
  """
  targets = tuple(targets)
  n_mod = 0
  for layer in gpt_model.gpt_layers:
    attn = layer.self_attention
    if 'query' in targets:
      attn.query = LoRALinear(attn.query, rank, alpha, dropout); n_mod += 1
    if 'key' in targets:
      attn.key = LoRALinear(attn.key, rank, alpha, dropout); n_mod += 1
    if 'value' in targets:
      attn.value = LoRALinear(attn.value, rank, alpha, dropout); n_mod += 1
    if 'attn_out' in targets:
      layer.attention_dense = LoRALinear(layer.attention_dense, rank, alpha, dropout); n_mod += 1
    if 'mlp' in targets:
      layer.interm_dense = LoRALinear(layer.interm_dense, rank, alpha, dropout); n_mod += 1
      layer.out_dense = LoRALinear(layer.out_dense, rank, alpha, dropout); n_mod += 1

  trainable = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in gpt_model.parameters())
  return n_mod, trainable, total
