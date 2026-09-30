#!/bin/bash
# Cut the unclipped annotated Stage 2-5 seizures as new labelled clips (cut_new_clips.py), CPU only.
# Every stage is resumable; outputs go only to $EEG_ROOT/output/ttg_newclips/.
#
#   bash submit_newclips.sh plan       # eeg env, here (openpyxl, cv2 and mne headers; ~30 s once paged in)
#   bash submit_newclips.sh motion     # scavenger array: one full decode per camera file (80), waits
#   bash submit_newclips.sh cut        # scavenger: ffmpeg copy cut + crop encode + info.txt, waits
#   bash submit_newclips.sh eeg        # eeg env, here: eeg.edf (mne 1.12.1 + edfio 0.4.16)
#   bash submit_newclips.sh analyze    # scavenger: activity + measure + decide (+ recut guard), waits
#   bash submit_newclips.sh sanity     # eeg env, here: line length + the TCN detector on CPU
#   bash submit_newclips.sh verify     # scavenger: verify + report, waits
#   ONLY=3,7 bash submit_newclips.sh motion   # re-run selected decode jobs
#
# The eeg env stages run in the calling shell (the interactive allocation) on purpose: on a cold scavenger node the
# eeg env takes 10-50 min just to import (rawdata/README.md); here it is paged in. Base stages go to scavenger.
#   bash submit_newclips.sh context    # eeg env, here: window-level detector context for the low-P clips
#   bash submit_newclips.sh report     # base, here: manifest.csv, summary.txt/json, integration.txt, items.csv
# What was run on 2026-09-29/30: plan, eeg, sanity and context in this allocation with $PYE; motion (array job
# 16662945) and analyze (activity + measure + decide + recut guard, job 16663191) on scavenger through this script;
# cut, verify, a re-run of decide (weak-only clusters, file fallback) and report in this allocation with $PY.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/work/mech-ai/alloy/miniconda3/bin/python}"                      # numpy / scipy, ffmpeg in its bin
PYE="${PYE:-/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python}"       # openpyxl, cv2, mne, edfio, torch
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/ttg_newclips"
mkdir -p "$OUT/logs"
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue --export=ALL"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
STAGE="${1:-help}"
S="$HERE/cut_new_clips.py"

wait_job() {   # sbatch --wait returns at once inside an interactive allocation, so poll sacct (squeue can come
  local j="${1%%;*}"   # back empty for a moment while an array is being scheduled)
  echo "job $j"
  sleep 30
  while env $UNSET sacct -j "$j" -n -X --format=State 2>/dev/null | grep -q -E "PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING"; do
    sleep 30
  done
}

case "$STAGE" in
  plan)
    $PYE "$S" plan 2>&1 | grep -v -e "Warning" -e "warn(" | tee "$OUT/logs/plan.out" ;;
  motion)
    N=$($PY -c "import json; print(len(json.load(open('$OUT/plan/jobs.json'))))")
    ARRAY="${ONLY:-0-$((N - 1))}"
    echo "motion array $ARRAY over $N camera files"
    JID=$($SB --parsable --array="$ARRAY" -c 8 --mem=8G -t 04:00:00 -J newclips_mot \
        -o "$OUT/logs/motion_%A_%a.out" \
        --wrap "$PY $S motion --index \$SLURM_ARRAY_TASK_ID --stride 100000 --threads 8")
    wait_job "$JID"
    grep -h -E "frames|cached|Error|failed" "$OUT"/logs/motion_*.out | tail -"$N" || true ;;
  cut)
    JID=$($SB --parsable -c 16 --mem=32G -t 06:00:00 -J newclips_cut -o "$OUT/logs/cut_%j.out" \
        --wrap "$PY $S cut --workers 8")
    wait_job "$JID"
    tail -20 "$(ls -t "$OUT"/logs/cut_*.out | head -1)" ;;
  eeg)
    $PYE "$S" eeg 2>&1 | grep -v -e "Warning" -e "warn(" | tee "$OUT/logs/eeg.out" ;;
  analyze)
    JID=$($SB --parsable -c 16 --mem=64G -t 06:00:00 -J newclips_ana -o "$OUT/logs/analyze_%j.out" \
        --wrap "$PY $S activity --workers 8 && $PY $S measure --workers 12 && $PY $S decide && $PY $S recut")
    wait_job "$JID"
    tail -40 "$(ls -t "$OUT"/logs/analyze_*.out | head -1)" ;;
  sanity)
    $PYE "$S" sanity --threads 8 2>&1 | grep -v -e "Warning" -e "warn(" | tee "$OUT/logs/sanity.out" ;;
  context)
    $PYE "$S" context --threads 8 2>&1 | grep -v -e "Warning" -e "warn(" | tee "$OUT/logs/context.out" ;;
  report)
    $PY "$S" report | tee "$OUT/logs/report.out" ;;
  verify)
    JID=$($SB --parsable -c 16 --mem=48G -t 06:00:00 -J newclips_ver -o "$OUT/logs/verify_%j.out" \
        --wrap "$PY $S verify --workers 12 && $PY $S report")
    wait_job "$JID"
    tail -40 "$(ls -t "$OUT"/logs/verify_*.out | head -1)" ;;
  *)
    sed -n '2,22p' "$0" ;;
esac
