#!/bin/bash
# Clip -> raw alignment on CPU scavenger nodes: one array task per labelled animal, then merge.
# Needs align/clips.csv and align/sample.csv first (align_clips.py inventory; align_clips.py sample).
# Resumable: a preempted/requeued task skips the clips already in units/<animal>.jsonl.
#   bash submit_align.sh             # all animals in sample.csv, waits, then merges
#   ONLY=3,7 bash submit_align.sh    # just these array indices (see: align_clips.py list)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# numpy/scipy from the miniconda3 base: the eeg env pages numpy in at ~7 min per process
PY="${PY:-/work/mech-ai/alloy/miniconda3/bin/python}"
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/raw_audit/align"
mkdir -p "$OUT/logs" "$OUT/units"
N=$($PY "$HERE/align_clips.py" list | wc -l)
ARRAY="${ONLY:-0-$((N - 1))}"
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "array $ARRAY over $N animals"
# sbatch --wait returned at once from inside an interactive allocation (2026-09-28), so poll squeue instead
JID=$($SB --parsable --array="$ARRAY" -c 4 --mem=16G -t 02:00:00 -J rawalign \
    -o "$OUT/logs/match_%A_%a.out" --export=ALL \
    --wrap "$PY $HERE/align_clips.py match --index \$SLURM_ARRAY_TASK_ID")
echo "job $JID"
while env $UNSET squeue -h -j "${JID%%;*}" 2>/dev/null | grep -q .; do sleep 30; done

$PY "$HERE/align_clips.py" merge > "$OUT/logs/merge.out" 2>&1 || echo "merge failed: see $OUT/logs/merge.out"
head -80 "$OUT/align_summary.txt"
