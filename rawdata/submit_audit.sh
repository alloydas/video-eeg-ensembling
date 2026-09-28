#!/bin/bash
# Run the raw-data audit on CPU scavenger nodes: one array task per animal folder, then merge.
# Resumable: a preempted/requeued task skips the files already in its units/<animal>.files.jsonl.
#   bash submit_audit.sh            # all folders, waits, then merges
#   ONLY=5,7 bash submit_audit.sh   # just these array indices (see: audit.py list)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python}"   # audit.py is stdlib-only
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/raw_audit"
mkdir -p "$OUT/logs" "$OUT/units"
N=$(ls -d /work/mech-ai/alloydas/EEG/Data/*/ | wc -l)
ARRAY="${ONLY:-0-$((N - 1))}"
# this shell may sit inside an interactive allocation: drop inherited SLURM_* so sbatch starts clean
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue"
export PYTHONDONTWRITEBYTECODE=1 PATH="/work/mech-ai/alloy/miniconda3/bin:$PATH"

echo "array $ARRAY over $N animal folders"
$SB --wait --array="$ARRAY" -c 8 --mem=24G -t 04:00:00 -J rawaudit \
    -o "$OUT/logs/unit_%A_%a.out" --export=ALL \
    --wrap "$PY $HERE/audit.py unit --index \$SLURM_ARRAY_TASK_ID --workers 8" || echo "array returned non-zero; merge will list missing units"

$SB --wait -c 2 --mem=16G -t 01:00:00 -J rawaudit_merge -o "$OUT/logs/merge_%j.out" --export=ALL \
    --wrap "$PY $HERE/audit.py merge"
tail -n +1 "$OUT/audit_summary.txt" | head -60
