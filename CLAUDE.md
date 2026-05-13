# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: GPT-2 from Scratch (NLP 2026-1 Final)

### Environment Setup

```bash
conda env create -f env.yml
conda activate nlp_final
```

Do not change package versions in `env.yml`. Only packages listed there are allowed for PART-I.

### Running Tests

```bash
# Test optimizer implementation
python optimizer_test.py

# Test GPT-2 model implementation (compares against HuggingFace GPT-2)
python sanity_check.py

# Train and evaluate sentiment classifier (SST + CFIMDB)
python classifier.py --fine-tune-mode last-linear-layer --use_gpu
python classifier.py --fine-tune-mode full-model --use_gpu

# Run paraphrase detection
python paraphrase_detection.py --use_gpu

# Run sonnet generation
python sonnet_generation.py --use_gpu
```

Adjust `--batch_size` based on GPU memory (default 8; SST can use 64 on 12GB GPU).

### Architecture

The project builds GPT-2 bottom-up:

1. **`modules/attention.py`** — `CausalSelfAttention`: multi-head causal self-attention. The `attention()` method (scores, masking, softmax, weighted sum) and reshape back to `[bs, seq_len, hidden_size]` are stubs.

2. **`modules/gpt2_layer.py`** — `GPT2Layer`: one transformer block. Uses pre-LayerNorm (LN before attention/FFN, unlike original GPT). `add()` applies dropout + residual (no LN). `forward()` is a stub.

3. **`models/gpt2.py`** — `GPT2Model`: stacks word + position embeddings → N `GPT2Layer`s → final LayerNorm. `embed()` and `hidden_state_to_token()` (weight-tied logits via `word_embedding.weight`) are stubs. `from_pretrained()` loads official HuggingFace GPT-2 weights by remapping them into this architecture.

4. **`models/base_gpt.py`** — `GPTPreTrainedModel`: weight initialization utilities.

5. **`config.py`** — `GPT2Config` (inherits `PretrainedConfig`): hyperparameters (vocab 50257, hidden 768, 12 layers, 12 heads, intermediate 3072).

6. **`optimizer.py`** — `AdamW`: stub; must implement 1st/2nd moment updates, bias correction, weight decay.

7. **`classifier.py`** — `GPT2SentimentClassifier`: fine-tunes GPT-2 for 5-class sentiment on SST/CFIMDB. Uses the last (non-padding) token hidden state. Supports `last-linear-layer` (frozen GPT) and `full-model` fine-tuning modes.

### PART-II Tasks

- **`paraphrase_detection.py`** — `ParaphraseGPT`: detect if two sentences are paraphrases (binary). Input is sentence-pair concatenated; output uses last-token hidden state → 2-class head. Core improvement task: go beyond the stub forward pass.

- **`sonnet_generation.py`** — `SonnetGPT`: autoregressive sonnet generation. Forward pass must return per-token logits (not just last token). Core improvement task: generation strategy (temperature, top-k/p sampling, etc.).

### Data

All datasets live in `data/`. Format: tab-separated CSV files.
- SST (`ids-sst-*`): 5-class sentiment
- CFIMDB (`ids-cfimdb-*`): binary sentiment
- Quora (`quora-*`): paraphrase detection pairs
- Sonnets (`sonnets.txt`, `sonnets_held_out*.txt`): Shakespeare sonnets for generation

Predictions written to `predictions/` directory.

### Key Constraints

- `sanity_check.py` validates `GPT2Model` output against HuggingFace's implementation with `atol=1e-1`. Pass this before training.
- Command-line arguments and their defaults must not be changed (project submission requirement).
- `GPT2Model.hidden_state_to_token()` must use weight tying: `logits = hidden_state @ word_embedding.weight.T`.
