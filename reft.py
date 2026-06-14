"""
ReFT (Representation Finetuning) — PART-II 확장 ③ (PEFT 비교용).

참고: Wu et al., "ReFT: Representation Finetuning for Language Models" (2024).

LoRA가 *가중치*에 저랭크 보정을 더한다면, ReFT(LoReFT)는 *은닉 표현*에 개입(intervention)한다.
선택한 레이어의 출력 은닉 상태 h 에 대해, 저차원 부분공간(R: 직교 행)에서만 표현을 편집한다:

    Φ(h) = h + Rᵀ ( W h + b − R h )

  - R ∈ ℝ^{r×d} : 직교(orthonormal) 행을 갖는 회전/투영. 학습되지만 직교성 유지.
  - W ∈ ℝ^{r×d}, b ∈ ℝ^r : 부분공간 안에서의 학습된 선형 편집.
  - 사전학습 가중치는 전부 동결, 개입 파라미터(R, W, b)만 학습.

본 구현은 모든 토큰 위치·선택 레이어에 동일 개입을 적용한 단순화 버전이다
(원 논문은 프롬프트 특정 위치에 적용). LoRA와 동일 조건에서 비교하기 위한 것이다.
"""

import torch
from torch import nn


class LoReftIntervention(nn.Module):
  """은닉 표현 h 에 대한 저랭크 직교 부분공간 개입."""

  def __init__(self, hidden_size: int, rank: int):
    super().__init__()
    # R: 직교 행을 갖는 [r, d] 투영. 직교성을 학습 내내 유지하도록 parametrization 사용.
    rotate = nn.Linear(hidden_size, rank, bias=False)
    nn.init.orthogonal_(rotate.weight)
    self.rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
    # 부분공간 안의 학습된 편집 W h + b.
    self.learned_proj = nn.Linear(hidden_size, rank)

  def forward(self, h):
    R = self.rotate.weight             # [r, d] (직교 행)
    Rh = self.rotate(h)                # [.., r]
    edit = self.learned_proj(h)        # [.., r]
    return h + (edit - Rh) @ R         # [.., r] @ [r, d] = [.., d]


class ReftLayer(nn.Module):
  """기존 GPT2Layer를 감싸 출력 은닉 상태에 개입을 적용한다."""

  def __init__(self, base_layer, intervention):
    super().__init__()
    self.base_layer = base_layer
    self.intervention = intervention

  def forward(self, hidden_states, attention_mask):
    out = self.base_layer(hidden_states, attention_mask)
    return self.intervention(out)


def inject_reft(gpt_model, rank=4, layers=None):
  """GPT2Model의 선택 레이어 출력에 LoReFT 개입을 주입.

  layers=None 이면 전체 레이어에 적용. 사전학습 가중치는 동결한다.
  반환: (개입한 레이어 수, 학습 가능 파라미터 수, 전체 파라미터 수)
  """
  # 베이스 동결 → 이후 생성하는 개입 파라미터만 학습됨.
  for p in gpt_model.parameters():
    p.requires_grad = False

  hidden_size = gpt_model.config.hidden_size
  n = len(gpt_model.gpt_layers)
  if layers is None:
    layers = list(range(n))

  interventions = nn.ModuleList()
  for i in layers:
    interv = LoReftIntervention(hidden_size, rank)
    gpt_model.gpt_layers[i] = ReftLayer(gpt_model.gpt_layers[i], interv)
    interventions.append(interv)
  # 모델 이동(.to(device))·state_dict에 포함되도록 등록.
  gpt_model.reft_interventions = interventions

  trainable = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in gpt_model.parameters())
  return len(layers), trainable, total
