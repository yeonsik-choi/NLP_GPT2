# PART-II 확장 및 개선 — 발표 자료

> 지정 주제 프로젝트(GPT-2) PART-II. 평가지표: Paraphrase = **Accuracy**, Sonnet = **chrF**.
> 목표(§7.4): *머신러닝 연구자의 관점에서 모델 성능 향상 방안을 탐구한다.*
> 윤리 준수: 모든 학습·튜닝·평가에 **train/dev만** 사용, 공식 test는 미사용.

---

## 1. 기반 구현 (누락 코드 완성)

| 파일 | 구현 내용 |
|---|---|
| `paraphrase_detection.py` `forward()` | **Cloze 방식**: 마지막 토큰 hidden → `hidden_state_to_token`(weight tying)으로 **전체 vocab logit** 반환. 평가가 `argmax`를 BPE 토큰 id(`yes`=8505/`no`=3919)와 직접 비교하므로 2-class head가 아닌 cloze가 정답 설계. |
| `sonnet_generation.py` `forward()` | 각 위치별 전체 vocab logit `[bs, seq, vocab]` 반환 → 토큰 단위 언어모델 학습. |
| 버그 수정 | `train()` 들여쓰기 오류, 파일 I/O `encoding='utf-8'`(Windows cp949 크래시 방지) |

**베이스라인 결과**
- Paraphrase (full fine-tuning, 1 epoch): **dev accuracy 86.4%**
- Sonnet (10 epoch): train loss 4.80 → **3.87**, 14편 생성 완료

---

## 2. 확장 1 — LoRA (PEFT) 로 Paraphrase 미세조정  ★헤드라인

### 동기
- Full fine-tuning은 GPT-2 **약 1억 2,400만 파라미터 전체**를 갱신 → 메모리·연산 비용이 크고, 대규모 Quora(다수 에폭)에서 과적합 위험.
- §7.4가 직접 권장한 **LoRA** [Hu et al., 2021] 적용.

### 방법 (`lora.py`)
- 어텐션의 **query / value** 프로젝션에만 저랭크 보정항 추가:
  `W' = W(frozen) + (α/r)·B·A`,  A:[r,in], B:[out,r], **B=0 초기화**(시작 시 ΔW=0).
- 사전학습 가중치는 전부 **동결**, LoRA 행렬만 학습.

### 결과 (1 epoch, batch_size=16)

| 설정 | 학습 파라미터 | 비율 | 학습 속도 | **dev accuracy** |
|---|---|---|---|---|
| Full fine-tuning (baseline) | ~124.4 M | 100 % | 8.73 it/s | **86.4 %** |
| LoRA r=8, α=16 (lr 2e-4) | 294,912 | **0.235 %** | 13.39 it/s (**~1.5×**) | 81.7 % |
| LoRA r=16, α=32 (lr 3e-4) | 589,824 | 0.47 % | ~15 it/s (**~1.7×**) | **85.3 %** |

### 핵심 메시지 (1 epoch, batch 16)
- **학습 파라미터를 99.5 % 줄이고 ~1.7배 빠르게** 학습하면서 full fine-tuning(86.4 %)에 **1.1 %p 차(85.3 %)**까지 근접.
- 효율-정확도 **트레이드오프**를 정량적으로 입증.

### 추가 ablation (동일 3,000 step, batch 8) — 모듈·rank·ReFT 비교 (단일 시드)
| 설정 | 학습 파라미터 | 비율 | it/s | dev acc |
|---|---|---|---|---|
| **D: LoRA all-modules r16 (8,000 step)** | 2,654,208 | 2.079 % | 25.9 | **0.8186** |
| **B: LoRA r=32 (q/v)** | 1,179,648 | 0.935 % | 32.9 | **0.8011** |
| A: full fine-tuning (참조) | 125 M | 100 % | 14.7 | 0.8001 |
| C: ReFT r=8 (표현 개입) | 147,552 | 0.118 % | 18.5 | 0.7735 |
| C: ReFT r=4 | 73,776 | 0.059 % | 23.9 | 0.7684 |

### ★다중 시드 유의성 검증 (3 시드: 11711/42/1234, 동일 3,000 step)
| 설정 | 비율 | **mean ± std** | min~max |
|---|---|---|---|
| full fine-tuning | 100 % | **0.8019 ± 0.0013** | 0.8001~0.8031 |
| LoRA r=32 (q/v) | 0.94 % | 0.7887 ± **0.0145** | 0.7684~0.8011 |
| LoRA r=8 (q/v) | 0.235 % | 0.7769 ± 0.0011 | 0.7761~0.7784 |
| LoRA r=16 (q/v) | 0.47 % | 0.7768 ± 0.0091 | 0.7691~0.7896 |
| ReFT r=8 | 0.118 % | 0.7717 ± 0.0038 | 0.7664~0.7753 |
| LoRA r=4 (q/v) | 0.118 % | 0.7647 ± 0.0047 | 0.7580~0.7684 |

- **주장 수정**: 단일 시드의 "r=32 ≈ full FT(80.1 vs 80.0)"는 r32의 **최량 시드**였음. 3-시드 평균 **78.9 % vs 80.2 %**로 full FT가 평균 1.3 %p 높고 **2/3 시드 우세**. → PEFT의 가치는 *정확도 동급*이 아니라 **효율**(99 % 적은 파라미터·2.2배 속도).
- **노이즈 확증**: mid-rank r8 vs r16은 Δ≈0(0.7769 vs 0.7768) → 차이는 시드 노이즈.
- **rank 효과**: r4→r32는 +2.4 %p(3/3 승)지만 중간 구간은 평탄. 용량 확대 시 **분산도 증가**(r16/r32 std 0.009~0.015 ≫ full/r4/r8).
- **모듈 ablation 3-시드**: "key 추가 하락"은 단일 시드 함정(최악 시드)으로 판명 — q/k/v 평균은 q/v와 노이즈 내 겹침. 모듈 확대(q/v→all +1.0 %p, 2/3, 노이즈성)는 rank 확대보다 약하고 덜 견고 → **"rank > 모듈" 방향 유지**.
- **α·dropout sweep 3-시드**: α(scaling) 키울수록 단조·견고 개선(α8→64: 76.6→78.9 %, 3/3). **🔑 α=64/r=8(0.235 %)이 r=32(0.935 %)와 동급** → rank 이점의 상당 부분은 **유효 업데이트 크기(α/r)** 효과, α가 더 값싼 지렛대. dropout은 짧은 예산(0.17 epoch)에서 무익(미세 하락).
- **α×lr 2D sweep 3-시드**: α와 lr은 곱(유효 업데이트)으로 환원되지 않는 **별개 축**(iso-product 비교 Δ=−0.68 %p, 매우 견고). 격자 최고 α=64/lr=4e-4 = **79.4 %**로 r=32를 **1/4 파라미터로 추월**. 공격적일수록 분산↑.
- **예제 단위 McNemar(dev 40K) + 학습 곡선**: r32 최량 시드조차 full FT와 **유의차 없음**(p=0.54) 확증. 반면 시드 단위 노이즈였던 r16 vs r8이 예제 단위로는 유의(p=1.2e-5) → **'단일-시드 유의성의 함정'**(두 검정은 서로 다른 질문). 학습 곡선: LoRA가 초기 적응이 빠르고 세 설정 모두 step ~2,500에서 ~0.80 수렴.
- **🔑 1-epoch 완주 수렴 비교 3-시드** (full vs r32 vs α64/r8, 17,688 step): full FT **86.0 %±0.8** > r32 84.3 %±0.2 > α64 83.8 %±0.5 — full FT가 **전 시드 우세** → "PEFT는 충분히 학습하면 따라잡는다" 가설 기각(동일-step의 1.3 %p 열위가 수렴에서 1.7 %p로 유지·확대). 동일-step에서 r32와 동급이던 α64/r8이 수렴에선 뒤짐 → **α/r 효과는 초기 적응 가속에 국한**, 수렴 상한은 실제 rank 용량이 결정. 분산은 **역전**(r32 std 0.0015 vs full 0.0077) → '용량↑→분산↑'은 과소적합 영역 한정 현상.

![PEFT 효율-정확도](figures/fig_peft_efficiency.png)

![수렴 비교](figures/fig_convergence.png)
- **ReFT 견고**: LoRA r8 대비 모든 시드에서 일관되게 0.5 %p 아래(절반 파라미터).
- **확장 학습(D)의 "추월"(81.9 %)은 3-시드에서 강등**: 평균 80.6 %±1.3으로 full FT 동일-step(80.2 %±0.1)과 **노이즈 내 동급**(최량 시드만 추월, 최악 78.8 %) — **네 번째 단일 시드 함정**. 공격적 설정(all-modules+lr 3e-4)의 고분산이 원인("강도↑→분산↑" 패턴). 반면 **§4.4 r16 1-epoch의 "1.1 %p 근접"은 3-시드 확증**(84.9 %±0.3 vs 86.0 %, 저분산).
- **현실적 제약 + 혼합정밀**: full FT batch 16에서 RAM spill(~20배 저하) 관측 → 후속 측정으로 원인이 **GPU 사전 점유**(실제 요구 ~3.9GB)임을 규명. **bf16은 정확도 손실 없이 +53% 가속**(batch16 9.43→14.41 it/s), bf16 batch16 ≈ fp32 batch8 속도. PEFT는 옵티마이저 상태가 작아 메모리 우위.

---

## 3. 확장 2 — Sonnet 디코딩 전략의 chrF 정량 비교

### 동기
- §7.3.2: `generate()` 개선 권장. 평가지표가 **chrF**(문자 n-gram F-score)이므로, **재학습 없이** 디코딩 전략만으로 점수를 올릴 수 있는지 탐구.

### 방법 (`sonnet_decode_experiment.py`)
- 공식 test는 미사용 → **학습 소네트 중 마지막 14편을 자체 held-out**으로 분리.
- 각 dev 소네트의 **첫 3줄을 조건**으로 나머지를 생성, 생성문 전체를 원본과 chrF 비교(설정 간 동일 시드).
- 동일 체크포인트(10 epoch 모델)로 8개 디코딩 설정 sweep.

### 결과 (dev chrF, 높을수록 좋음)

| 순위 | 디코딩 전략 | dev chrF |
|---|---|---|
| 1 | temperature=1.2 + top-p=0.9 (기본값) | **42.56** |
| 2 | top-p=0.9, temp=0.9 | 41.08 |
| 3 | temperature=1.0 | 40.83 |
| 4 | temperature=0.7 / top-k=40(temp0.9) | 39.98 |
| 6 | top-k=40 + top-p=0.9 | 39.88 |
| 7 | top-p=0.9, temp=0.7 | 38.73 |
| 8 | **greedy** | **31.68** |
*(위 표는 단일 시드 11711)*

### ★다중 시드 chrF + beam search (3 시드: 11711/42/1234)
| 전략 | 종류 | **mean ± std** | 시드별 |
|---|---|---|---|
| temp=1.2 + top-p=0.9 (기본값) | 샘플링 | **42.06 ± 0.36** | 42.56/41.81/41.79 |
| temp=1.0 | 샘플링 | 41.29 ± 0.34 | |
| top-p=0.9 temp=0.9 | 샘플링 | 41.12 ± 0.09 | 매우 안정 |
| best + rep_penalty=1.2 | 샘플링 | 39.79 ± **1.30** | 38.00/40.36/41.02 |
| beam=3 | 결정론적 | 32.19 | (시드 무관) |
| beam=5 | 결정론적 | 31.86 | (시드 무관) |
| beam=1 (greedy) | 결정론적 | 31.70 | (시드 무관) |

### 핵심 메시지
- **greedy(31.70) → 최적 샘플링(42.06 평균) = +10.4 chrF, 3-시드 견고**. 최적 설정 최저 시드(41.79)도 모든 beam/greedy보다 +9.6 위.
- **beam search 음성 결과 (신규)**: beam=3은 greedy 대비 +0.5뿐, beam=5는 하락하며 샘플링에 **~10점 뒤짐** → **"우도 최대화 ≠ 생성 품질"**(Holtzman et al. 2020). 결정론적 탐색은 창작 도메인에 부적합.
- **repetition penalty 음성 결과 — 진짜이나 단일 시드가 과장**: 단일 38.00은 최악 시드, 평균 39.79±1.30(best 대비 -2.3). 여전히 음성·**최대 분산**.
- **퇴화 지표로 메커니즘 입증** (distinct-n·rep-4, 14편 평균): beam=3은 고유 unigram **10 %**·4-gram의 **86 %가 반복**(전형적 퇴화)이라 31~32점에 머묾. rep_penalty는 distinct-1 **0.998**로 '과교정'(반대 방향으로 원본에서 멀어짐). 최적 샘플링(distinct-1 0.75, rep-4 0)이 원본 다양성에 가장 근접 → **chrF는 원본 수준 다양성과의 일치를 보상**.
- 결론: temperature 1.0~1.2 + nucleus(top-p 0.9)가 최적이자 저분산. 제공 기본값이 잘 설정됨을 다중 시드로 검증.

![디코딩 chrF와 퇴화 지표](figures/fig_sonnet_chrf.png)

---

## 4. 종합 결론

1. **Cloze 재구성**으로 GPT-2를 분류기로 전환 → Quora 1-epoch 86.4 %.
2. **LoRA**: 1-epoch에서 0.47 %·1.7배 속도로 full FT와 1.1 %p 차(85.3 %). **3-시드 검증**: r=32(0.94 %)는 평균 78.9 %±1.5로 full FT(80.2 %±0.1)에 *최량 시드에서만* 도달(평균 1.3 %p 열위, full FT 2/3 승) → PEFT의 가치는 **효율**. 확장 학습의 "추월"(81.9 %)도 3-시드에서 **동급으로 강등**(80.6 %±1.3). **1-epoch 완주 3-시드**에서도 full FT(86.0 %)가 r32(84.3 %)·r16(84.9 %)을 전 시드에서 앞서 "충분히 학습하면 따라잡는다" 가설 기각 — 단 LoRA는 2.1배 빠름(12분 vs 26분/epoch)이고 §4.4 r16의 1.1 %p 근접은 견고.
3. **mid-rank 차이는 노이즈로 확증**, 용량↑ 시 분산↑(단 **과소적합 한정** — 1-epoch 수렴에선 역전) / **α=64/r8이 r32와 동급(1/4 파라미터)** → rank 이점=유효 업데이트 크기 효과, 단 **수렴에선 r32 우세** → α/r 효과는 초기 적응 가속에 국한, dropout 무익 / α×lr은 **별개 축**(격자 최고 79.4 %가 r32 추월) / McNemar로 r32≈full 확증 + **'단일-시드 유의성의 함정'** 발견 / **ReFT는 절반 파라미터로 LoRA r8에 일관되게 0.5 %p 근접**.
4. full FT **RAM spill(~20배)** 원인 규명(GPU 사전 점유, 실제 요구 ~3.9GB) + **bf16 +53% 가속(정확도 무손실)** + PEFT 메모리 우위 실측.
5. **디코딩 전략**: 재학습 없이 chrF **+10.4(3-시드 견고)**. **beam search는 샘플링에 ~10점 뒤짐**("우도≠품질"), repetition penalty는 **해로움(음성 결과, 고분산)**. 퇴화 지표로 메커니즘 입증(beam rep-4 86 % 반복 vs rep_penalty 과교정 distinct-1 0.998).

### 완료된 확장 (당초 향후 과제였던 항목)
- ✅ LoRA 대상 모듈 확대(Q/K/V+attn_out+MLP) ablation — `lora.py`, `lora_experiments.py`
- ✅ rank ablation(r=4/8/16/32), 확장 학습(8,000 step)
- ✅ ReFT(LoReFT) 직접 구현·비교 — `reft.py`
- ✅ Sonnet repetition penalty 추가·평가
- ✅ **3-시드 유의성 검증(PEFT)** — `multiseed_experiments.py`(rank), `module_multiseed.py`(모듈) (r32 "동급"·"key 하락" 단일 시드 함정 규명, mid-rank 노이즈 확증)
- ✅ **Sonnet beam search 구현·비교 + 디코딩 3-시드 chrF** — `sonnet_generation.py`(generate_beam), `sonnet_decode_multiseed.py`
- ✅ **혼합정밀(bf16) 메모리·속도 실측 + §4.7 spill 원인 규명** — `amp_experiment.py`
- ✅ **LoRA α·dropout 정밀 sweep(3-시드)** — `lora_hparam_sweep.py` (α가 rank를 1/4 파라미터로 대체, dropout 무익)
- ✅ **α×lr 2D sweep(3-시드)** — `alpha_lr_sweep.py` (α와 lr은 별개 축, 격자 최고 79.4 %)
- ✅ **예제 단위 McNemar 검정 + 학습 곡선** — `stat_analysis.py`, `learning_curve.py` ('단일-시드 유의성의 함정')
- ✅ **생성 퇴화 지표(distinct-n·rep-4) 정량화** — `degeneration_metrics.py` (beam 퇴화·rep_penalty 과교정 입증)
- ✅ **1-epoch 완주 수렴 비교(3-시드)** — `multiepoch_experiment.py` ("따라잡는다" 가설 기각, α/r=초기 가속, 분산 역전)
- ✅ **확장 학습 D·§4.4 r16 헤드라인 3-시드 승격** — `d8000_multiseed.py`, `paraphrase_detection.py --seed` (D "추월"은 단일 시드 함정으로 강등, r16 근접은 확증 — **모든 헤드라인 비교가 다중 시드 검증 완료**)

### 남은 향후 방향
- 시드 수↑ 후 교차-시드 형식 검정, 1 epoch 초과 다중 에폭(dropout이 의미를 갖는 과적합 영역), 더 큰 rank·α·lr 동시 상향의 안정성 경계 탐색, fp16+GradScaler·gradient checkpointing, Sonnet 운율 제약 디코딩.

### 참고문헌
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Wu et al., *ReFT: Representation finetuning for language models*, 2024.
- Keskar et al., *CTRL: A Conditional Transformer Language Model*, 2019.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020.
- Popović, *chrF: character n-gram F-score for automatic MT evaluation*, 2015.

---

### 재현 명령어
```bash
# Paraphrase — full fine-tuning baseline
python paraphrase_detection.py --use_gpu --epochs 1 --batch_size 16

# Paraphrase — LoRA (PEFT)
python paraphrase_detection.py --use_gpu --epochs 1 --batch_size 16 --lr 2e-4 \
    --use_lora --lora_rank 8 --lora_alpha 16

# Paraphrase — PEFT ablation (모듈/rank/ReFT, 동일 step, 단일 시드)
python lora_experiments.py --use_gpu --batch_size 8 --max_steps 3000 --full_steps 8000

# Paraphrase — 다중 시드 유의성 검증 (rank / 모듈 / α·dropout / α×lr)
python multiseed_experiments.py --use_gpu --batch_size 8 --max_steps 3000 --seeds 11711 42 1234
python module_multiseed.py --use_gpu --batch_size 8 --max_steps 3000 --seeds 11711 42 1234
python lora_hparam_sweep.py --use_gpu --batch_size 8 --max_steps 3000 --seeds 11711 42 1234
python alpha_lr_sweep.py --use_gpu --batch_size 8 --max_steps 3000 --seeds 11711 42 1234

# Paraphrase — 예제 단위 McNemar 검정 + 학습 곡선
python stat_analysis.py --use_gpu --batch_size 8 --max_steps 3000
python learning_curve.py --use_gpu --batch_size 8 --max_steps 3000

# Paraphrase — 1-epoch 완주 수렴 비교 (3-시드)
python multiepoch_experiment.py --use_gpu --batch_size 16 --epochs 1 --seeds 11711 42 1234

# Paraphrase — 확장 학습 D 3-시드 + §4.4 r16 헤드라인 시드 승격
python d8000_multiseed.py --use_gpu --batch_size 8 --max_steps 8000 --seeds 11711 42 1234
python paraphrase_detection.py --use_gpu --epochs 1 --batch_size 16 --lr 3e-4 \
    --use_lora --lora_rank 16 --lora_alpha 32 --seed 42 \
    --para_dev_out predictions/para-dev-lora16-s42.csv --para_test_out predictions/para-test-lora16-s42.csv

# Paraphrase — 혼합정밀(bf16) 메모리·속도 측정
python amp_experiment.py --use_gpu --steps 300

# Sonnet — 학습 + 제출 파일 생성
python sonnet_generation.py --use_gpu --epochs 10 --batch_size 8

# Sonnet — 디코딩 전략 chrF 비교 (단일 시드)
python sonnet_decode_experiment.py --use_gpu

# Sonnet — 디코딩 다중 시드 + beam search 비교
python sonnet_decode_multiseed.py --use_gpu --seeds 11711 42 1234

# Sonnet — 생성 퇴화 지표 (distinct-n·rep-4)
python degeneration_metrics.py --use_gpu
```
