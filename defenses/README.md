# Defense Mechanisms

Each defense targets a specific attack in the PoisonedRAG pipeline.

| Attack | Best Defense | File |
|---|---|---|
| LM_targeted | Combined (Rationale + Voting) | `combined_defense.py` |
| HotFlip | Perplexity Filter | `perplexity_filter.py` |
| Corpus Poisoning | Rationale-Based Denoising | `rationale_denoising.py` |
| Prompt Injection | Regex Instruction Filter | `regex_instruction_filter.py` |

---

## Proven Results (GPT-4, NQ dataset)

| Attack | Baseline ASR | Defended ASR | Reduction |
|---|---|---|---|
| LM_targeted | 1.0 | 0.2 | -80% |
| HotFlip | 1.0 | 0.5 | -50% |
| Corpus Poisoning | 0.75 | 0.25 | -67% |
| Prompt Injection | 1.0 | 0.0 | -100% |

---

## Attack Only (No Defense)

**LM_targeted**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method LM_targeted --M 5 --repeat_times 2 \
  --random_targets True --seed 42
```

**HotFlip** (slow on CPU — ~10 min per question)
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method hotflip --M 2 --repeat_times 2
```

**Corpus Poisoning**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method corpus_poisoning --M 5 --repeat_times 2 \
  --random_targets True --seed 42
```

**Prompt Injection**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method prompt_injection --M 5 --repeat_times 2 \
  --random_targets True --seed 42
```

**No Attack (baseline)**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method None --M 5 --repeat_times 2 \
  --random_targets True --seed 42
```

---

## Attack + Defense

**LM_targeted + Combined Defense** (Rationale + Voting)
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method LM_targeted --M 5 --repeat_times 2 \
  --random_targets True --seed 42 \
  --defense combined \
  --defense_output results/defense_results/lm_targeted_defense.json
```

**HotFlip + Perplexity Filter**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method hotflip --M 2 --repeat_times 2 \
  --defense perplexity_filter \
  --defense_output results/defense_results/hotflip_defense.json
```

**Corpus Poisoning + Rationale Denoising**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method corpus_poisoning --M 5 --repeat_times 2 \
  --random_targets True --seed 42 \
  --defense rationale_denoising \
  --defense_output results/defense_results/corpus_poisoning_defense.json
```

**Prompt Injection + Regex Instruction Filter**
```bash
/opt/anaconda3/envs/PoisonedRAG/bin/python main.py \
  --eval_dataset nq --model_name gpt4 \
  --attack_method prompt_injection --M 5 --repeat_times 2 \
  --random_targets True --seed 42 \
  --defense regex_instruction_filter \
  --defense_output results/defense_results/prompt_injection_defense.json
```

---

## Available Defense Names (--defense flag)

| Flag Value | Defense | Best Against |
|---|---|---|
| `combined` | Rationale Denoising + Isolate-then-Aggregate Voting | LM_targeted |
| `perplexity_filter` | GPT-2 perplexity scoring | HotFlip |
| `rationale_denoising` | LLM reasoning per document | Corpus Poisoning |
| `regex_instruction_filter` | Pattern matching on injection commands | Prompt Injection |
| `multimodel_reranking` | BM25 + dense re-ranking | LM_targeted (partial) |

---

## Output Format

Results saved to `--defense_output` path as JSON:

```json
{
  "attack_method": "LM_targeted",
  "defense": "combined",
  "asr_mean": 1.0,
  "asr_defended_mean": 0.2,
  "asr_reduction": 0.8,
  "asr_per_repeat": [1.0, 1.0],
  "asr_defended_per_repeat": [0.2, 0.2]
}
```

`asr_reduction` = baseline ASR − defended ASR. Higher is better for the defense.

---

## Defense Details

### 1. Combined Defense (`combined_defense.py`)
**Best against:** LM_targeted

Two-stage defense:
1. **Rationale step** — LLM evaluates each doc using its own world knowledge, marks each TRUST / DISCARD / UNCERTAIN. Warns the LLM that all docs may be poisoned and not to trust majority agreement.
2. **Isolation + Voting step** — LLM answers independently from each trusted doc, then keyword frequency vote picks the winner.

The rationale answer takes priority over the vote — it uses the LLM's world knowledge directly when all docs are discarded.

Achieved **-80% ASR reduction** on LM_targeted with GPT-4 (M=5).

---

### 2. Perplexity Filter (`perplexity_filter.py`)
**Best against:** HotFlip

HotFlip flips tokens to fool the retriever but breaks natural language fluency. GPT-2 scores each retrieved doc — docs above the perplexity threshold are removed. When all docs are filtered, the LLM answers from its own knowledge with no context (no poisoned docs at all).

Key parameters:
- `threshold=150.0` — perplexity cutoff (HotFlip docs score 226-829, clean docs score 20-100)
- `model_name="gpt2"` — lazy-loaded on first use

Achieved **-50% ASR reduction** on HotFlip with GPT-4 (M=2).

---

### 3. Rationale-Based Denoising (`rationale_denoising.py`)
**Best against:** Corpus Poisoning

Corpus poisoning produces fluent, well-written fake docs that pass perplexity and regex checks. The LLM is instructed to reason about each doc's trustworthiness using its own knowledge before answering. Uses a skeptical prompt that explicitly warns the LLM that all docs may be poisoned.

No extra parameters — uses the same LLM instance as the main pipeline.

Achieved **-67% ASR reduction** on Corpus Poisoning with GPT-4 (M=5).

---

### 4. Regex Instruction Filter (`regex_instruction_filter.py`)
**Best against:** Prompt Injection

Prompt injection embeds explicit override commands in retrieved docs (SYSTEM OVERRIDE, [CRITICAL], AUTHORITATIVE SOURCE, VERIFIED FACT, etc.). These are directly detectable via regex. All matching docs are removed. When all docs are filtered, the LLM answers from its own knowledge.

Key parameters:
- `sanitize=False` — remove whole doc (default) vs. strip just the matched pattern
- `custom_patterns=[]` — add your own regex patterns

Achieved **-100% ASR reduction** on Prompt Injection with GPT-4 (M=5).

---

### 5. Multi-Model Re-ranking (`multimodel_reranking.py`)
**Against:** LM_targeted (limited effectiveness)

Re-ranks retrieved docs using BM25 alongside Contriever scores. Docs with very low BM25 scores relative to the best BM25 doc are flagged as suspicious. 

Note: LM_targeted adversarial docs are keyword-rich and fluent, so BM25 scores them similarly to clean docs. This defense works better when adversarial docs are semantically similar but lexically different from the query. Superseded by the Combined Defense for LM_targeted.

Key parameters:
- `alpha=0.5` — weight split between dense and BM25 scores
- `bm25_min_ratio=0.1` — docs scoring below 10% of max BM25 are flagged

---

## How It Works — Pipeline Flow

```
Retrieved top-k docs (may include adversarial docs)
        |
        ├── regex_instruction_filter  → removes injection pattern docs
        |
        ├── perplexity_filter         → removes high-perplexity (token-flipped) docs
        |
        ├── multimodel_reranking      → demotes low-BM25 docs
        |
        ├── rationale_denoising       → LLM reasons about each doc
        |
        └── combined                  → rationale + isolation + voting
                |
                └── Final defended answer
```

Each defense operates independently. The `--defense` flag selects one.
Results include both baseline ASR and defended ASR for direct comparison.
