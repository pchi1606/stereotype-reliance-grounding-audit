#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# Downloads the public source datasets for full regeneration.
# The bundled data/ and outputs/ are sufficient for fast verification
# (bash reproduce.sh); this script is only needed to regenerate from scratch.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data_big

echo "Downloading GlobalOpinionQA (Anthropic/llm_global_opinions)..."
python -c "
from datasets import load_dataset
ds = load_dataset('Anthropic/llm_global_opinions')
ds.save_to_disk('data_big/global_opinions')
print('saved')
"

echo "Downloading WVS Wave 7 SFT (oxford-llms/world_values_survey_2017_2022_sft)..."
python -c "
from datasets import load_dataset
ds = load_dataset('oxford-llms/world_values_survey_2017_2022_sft')
ds.save_to_disk('data_big/wvs_sft')
print('saved')
"

echo "Downloading Qwen3-1.7B Q4_K_M GGUF (bartowski)..."
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('bartowski/Qwen_Qwen3-1.7B-GGUF', 'Qwen_Qwen3-1.7B-Q4_K_M.gguf', local_dir='data_big')
print('saved to', p)
"
echo "Done. See the GitHub repository for the full generation pipeline."
