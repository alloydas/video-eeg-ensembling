#!/bin/bash
# Re-cut the labelled clips whose video is from the wrong time (recut_clips.py), on CPU scavenger nodes.
# Every stage is resumable; outputs go only to $EEG_ROOT/output/ttg_recut/.
#   bash submit_recut.sh motion     # plan (here), then one array task per camera file (full decode), waits
#   bash submit_recut.sh analyze    # activity + measure + calibrate (4 transforms) + decide on one node, waits
#   bash submit_recut.sh recut      # recut + repro + verify + report on one node, waits
#   bash submit_recut.sh verify     # verify + report only (after a change to the checks)
#   bash submit_recut.sh all        # motion, analyze, recut in order
#   ONLY=3,7 bash submit_recut.sh motion   # re-run selected decode jobs (indices printed by `plan`)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# numpy/scipy from the miniconda3 base: the eeg env pages numpy in at ~7 min per process (rawdata/README.md)
PY="${PY:-/work/mech-ai/alloy/miniconda3/bin/python}"
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/ttg_recut"
mkdir -p "$OUT/logs"
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue --export=ALL"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
STAGE="${1:-all}"

wait_job() {   # sbatch --wait returns at once inside an interactive allocation, so poll squeue
  local j="${1%%;*}"
  echo "job $j"
  while env $UNSET squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 30; done
}

if [[ "$STAGE" == motion || "$STAGE" == all ]]; then
  $PY "$HERE/recut_clips.py" plan | tee "$OUT/logs/plan.out"
  N=$($PY -c "import json; print(len(json.load(open('$OUT/plan/jobs.json'))))")
  ARRAY="${ONLY:-0-$((N - 1))}"
  echo "motion array $ARRAY over $N camera files"
  JID=$($SB --parsable --array="$ARRAY" -c 8 --mem=8G -t 03:00:00 -J recut_mot \
      -o "$OUT/logs/motion_%A_%a.out" \
      --wrap "$PY $HERE/recut_clips.py motion --index \$SLURM_ARRAY_TASK_ID --threads 8")
  wait_job "$JID"
  grep -h -E "frames|cached|Error|failed" "$OUT"/logs/motion_*.out | tail -"$N" || true
fi

if [[ "$STAGE" == analyze || "$STAGE" == all ]]; then
  JID=$($SB --parsable -c 8 --mem=32G -t 04:00:00 -J recut_ana -o "$OUT/logs/analyze_%j.out" \
      --wrap "$PY $HERE/recut_clips.py activity --workers 8 && $PY $HERE/recut_clips.py measure && \
              (for c in 'rank 0' 'log 0' 'log1p 0' 'rank 61'; do set -- \$c; \
                 RECUT_MOT_TRANSFORM=\$1 RECUT_HP_WIN_S=\$2 $PY $HERE/recut_clips.py calibrate \
                 > $OUT/logs/calibrate_\$1_hp\$2.out 2>&1 & done; wait) && \
              $PY $HERE/recut_clips.py decide")
  wait_job "$JID"
  tail -30 "$(ls -t "$OUT"/logs/analyze_*.out | head -1)"
fi

if [[ "$STAGE" == recut || "$STAGE" == all ]]; then
  JID=$($SB --parsable -c 8 --mem=32G -t 04:00:00 -J recut_cut -o "$OUT/logs/recut_%j.out" \
      --wrap "$PY $HERE/recut_clips.py recut --workers 4 && $PY $HERE/recut_clips.py repro && \
              $PY $HERE/recut_clips.py verify --workers 4 && $PY $HERE/recut_clips.py report")
  wait_job "$JID"
  tail -40 "$(ls -t "$OUT"/logs/recut_*.out | head -1)"
fi

if [[ "$STAGE" == verify ]]; then
  JID=$($SB --parsable -c 8 --mem=32G -t 02:00:00 -J recut_ver -o "$OUT/logs/verify_%j.out" \
      --wrap "$PY $HERE/recut_clips.py verify --workers 4 && $PY $HERE/recut_clips.py report")
  wait_job "$JID"
  tail -40 "$(ls -t "$OUT"/logs/verify_*.out | head -1)"
fi
