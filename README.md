# Reproduction package — Which Grounding Prevents Stereotype Reliance? A Three-Arm Audit of LLM-Simulated Regional Humans

**Author:** Prabhu Rangaraju (Independent Researcher)

Reproduces the analysis and verification of *"Which Grounding Prevents
Stereotype Reliance? A Three-Arm Audit of LLM-Simulated Regional Humans"*
(manuscript in preparation; target venue ACM FAccT 2027).

## One command (fast verification, <2 minutes)

From this directory:

```bash
bash reproduce.sh
```

This installs dependencies, re-runs the preregistered analysis on the bundled
4,672 generation records, and verifies the headline numbers against
`expected/values.json` (PASS/FAIL per check).

## Contents

- `data/` — items.json (12 WVS items with GlobalOpinionQA human distributions),
  plan_multi.json (336 prompts: randomized item orders and arm-2 personas)
- `code/` — analyze.py, metrics.py (preregistered metrics and hypothesis tests)
- `outputs/` — generations.jsonl (4,672 records: 336 multi-item prompts × 12
  answers + 640 single-item pilot records), results.json
- `expected/values.json` — headline numbers for verification
- `verify.py` — PASS/FAIL checker
- `paper.pdf` — the manuscript
- `fetch_data.sh` — downloads public source datasets (for full regeneration)

Note on `outputs/results.json`: the key `arm0_gb_match` (0.375) is the
Great Britain modal-profile match rate over full-experiment neutral
generations; the manuscript's pilot section reports the pilot-neutral value
(35.0%, 112/320) instead. Both are stored; neither is an error.

## Full regeneration (optional, 6–8 hours)

To regenerate the generations from scratch (requires the 1.2 GB Qwen3-1.7B
model and CPU inference via llama.cpp):

1. `bash fetch_data.sh` — downloads GlobalOpinionQA, the WVS individual-response
   corpus, and the model weights.
2. See the GitHub repository for the full generation pipeline
   (`code/generate_multi.py`, `code/build_plan_multi.py`).

## Data licenses

- GlobalOpinionQA (Durmus et al. 2023): public research dataset
  (https://huggingface.co/datasets/Anthropic/llm_global_opinions).
- WVS individual-response corpus: public HuggingFace redistribution
  (https://huggingface.co/datasets/oxford-llms/world_values_survey_2017_2022_sft);
  original source WVS Wave 7 (Haerpfer et al. 2022).
- Qwen3-1.7B weights: Qwen license
  (https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF).

## License

MIT (see LICENSE).
