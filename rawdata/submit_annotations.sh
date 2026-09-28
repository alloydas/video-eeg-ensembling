#!/bin/bash
# Run annotations.py on one CPU scavenger node: every unit (process pool), then the report.
# Resumable: finished units leave a cache file under $OUT/cache/ and are skipped on a rerun or requeue.
#   bash submit_annotations.sh                 # all units + report, waits
#   CMD=report bash submit_annotations.sh      # report only (cheap)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python}"
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/raw_audit/annotations"
mkdir -p "$OUT/logs" "$OUT/cache"
CMD="${CMD:-all}"
W="${WORKERS:-16}"
# this shell may sit inside an interactive allocation: drop inherited SLURM_* so sbatch starts clean
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
export PYTHONDONTWRITEBYTECODE=1 PATH="/work/mech-ai/alloy/miniconda3/bin:$PATH"
env $UNSET sbatch --wait -A mech-ai-scavenger -q scavenger -p scavenger --requeue \
    -c "$W" --mem=48G -t 03:00:00 -J rawannot -o "$OUT/logs/annot_%j.out" --export=ALL \
    --wrap "$PY $HERE/annotations.py $CMD --workers $W"
tail -n 40 "$(ls -t "$OUT"/logs/annot_*.out | head -1)"
