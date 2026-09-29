#!/bin/bash
# Re-score the EEG-gated grader with the re-cut clips (rescore_recut.py), on CPU scavenger nodes. No GPU.
# Outputs go only to $EEG_ROOT/output/ttg_recut/.
#   bash submit_rescore.sh prep      # cache + infer (fp32, 30 runs) + patch, one node, waits
#   bash submit_rescore.sh grid      # 6 joint_gate grids in parallel (2 recipes x orig/exclonly/patched), waits
#   bash submit_rescore.sh analyse   # compare (wrapper faithfulness) + analyse, waits
#   bash submit_rescore.sh all       # prep, grid, analyse in order
# What was actually run on 2026-09-29 (in the interactive allocation: the eeg env took 47 min to import on a cold
# node, so the work stayed where it was already paged in):
#   $PY rescore_recut.py cache infer patch --threads 8      # stopped at the equivalence gate (2 rows > 0.02, argmax ok)
#   /work/mech-ai/alloy/miniconda3/bin/python rescore_recut.py infer patch --accept_precision_noise   # numpy only
#   $PY rescore_recut.py grid compare analyse --jobs 6
# 'prep' below stops at the same gate; add --accept_precision_noise only after reading infer/equivalence.json.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python}"     # torch, cv2, sklearn 1.5.2
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/ttg_recut"
mkdir -p "$OUT/logs"
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue --export=ALL"
export PYTHONDONTWRITEBYTECODE=1
STAGE="${1:-all}"
S="$HERE/rescore_recut.py"

wait_job() {   # sbatch --wait returns at once inside an interactive allocation, so poll squeue
  local j="${1%%;*}"
  echo "job $j"
  while env $UNSET squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 30; done
}

if [[ "$STAGE" == prep || "$STAGE" == all ]]; then
  JID=$($SB --parsable -c 8 --mem=32G -t 04:00:00 -J rescore_prep -o "$OUT/logs/rescore_prep_%j.out" \
      --wrap "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY $S cache infer patch --threads 8")
  wait_job "$JID"
  tail -45 "$(ls -t "$OUT"/logs/rescore_prep_*.out | head -1)"
fi

if [[ "$STAGE" == grid || "$STAGE" == all ]]; then
  # 6 joint_gate runs (2 recipes x orig / exclonly / patched) as forked children of ONE process: the eeg env
  # takes 10-50 min to import on this filesystem, so it is imported once
  JID=$($SB --parsable -c 8 --mem=48G -t 04:00:00 -J rescore_grid -o "$OUT/logs/rescore_grid_%j.out" \
      --wrap "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY $S grid --jobs 6")
  wait_job "$JID"
  cat "$(ls -t "$OUT"/logs/rescore_grid_*.out | head -1)"
fi

if [[ "$STAGE" == analyse || "$STAGE" == all ]]; then
  JID=$($SB --parsable -c 8 --mem=32G -t 02:00:00 -J rescore_ana -o "$OUT/logs/rescore_analyse_%j.out" \
      --wrap "OMP_NUM_THREADS=8 $PY $S compare analyse")
  wait_job "$JID"
  tail -80 "$(ls -t "$OUT"/logs/rescore_analyse_*.out | head -1)"
fi
