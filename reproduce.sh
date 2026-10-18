#!/usr/bin/env bash
# Author: Prabhu Rangaraju
# One-command reproduction of the S11 stereotype-reliance audit analysis.
#
# Usage: bash reproduce.sh
# Run from this directory. Requires Python 3.10+.
#
# What it does:
#   1. installs Python dependencies (requirements.txt)
#   2. runs the analysis pipeline on the bundled generations
#      (code/analyze.py -> outputs/results.json)
#   3. verifies headline numbers against expected/values.json (PASS/FAIL)
#
# The bundled outputs/generations.jsonl contains all 4,672 records
# (336 multi-item prompts + 640 pilot records). To regenerate from scratch,
# see fetch_data.sh for the public data sources and the full pipeline
# in the GitHub repository.
#
# Expected runtime: <2 minutes on a modern laptop.
set -euo pipefail
cd "$(dirname "$0")"

echo "================================================================"
echo " STEP 1/3: dependencies"
echo "================================================================"
pip install -r requirements.txt || echo "WARNING: pip install failed; continuing."

echo "================================================================"
echo " STEP 2/3: analysis pipeline"
echo "================================================================"
PY=$(command -v python3 || command -v python)
$PY code/analyze.py > /tmp/s11_repro_analysis.log 2>&1
echo "analysis complete; see outputs/results.json"

echo "================================================================"
echo " STEP 3/3: verification"
echo "================================================================"
$PY verify.py
