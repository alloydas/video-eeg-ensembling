#!/bin/bash
#SBATCH --job-name=ttg-probe
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/probe_%j.out
#
# ==========================================================================================
#  TT-X3D Stage 2: frozen X3D-M features (sparse16 / dense / dense_shuffled) for every clip.
# ==========================================================================================
#
# PURPOSE
#   Runs grader/probe_extract.py on one GPU. Decoding (8 frame-exact seeks x 31 native frames
#   per clip from data_full) happens in 14 DataLoader workers; the GPU runs 17 X3D-M views
#   per clip in fp16 autocast. Output: $EEG_ROOT/output/ttg_probe/features.npz, consumed on
#   CPU by grader/probe_analysis.py (the pre-registered kill test). The extractor is
#   $GRADER_DIR/probe_extract.py (video-eeg-ensembling/grader; under sbatch "$0" is a spooled
#   copy, so $GRADER_DIR, else the checkout sbatch was run from, else the fixed default is
#   used) run with cwd = EEG_ROOT (env var,
#   default /work/mech-ai-scratch/alloy/EEG).
#
# PREEMPTION AND TIME LIMIT
#   Scavenger requeues preempted jobs (PreemptMode=REQUEUE, KillWait 30 s). Work is written
#   in 512-clip shards atomically, so a requeued job skips finished shards and loses at most
#   one partial shard. --time is explicit because the partition default is 4 h (8 h is ample:
#   CPU-timed decode is 0.25-0.62 s/clip, ~15 min over 14 workers). --signal=B:USR1@300
#   warns this shell 5 min before the limit; it stops the extractor and requeues the job
#   itself with `scontrol requeue` (a plain SIGTERM is not requeued here, so scancel works).
#   --retry_failed makes every (re)start re-extract rows that failed in finished shards, so
#   a transient NFS error does not become a permanent drop.
#
# USAGE
#   mkdir -p /work/mech-ai-scratch/alloy/EEG/logs/ttg      # SLURM will not create it
#   (or: bash grader/submit_ttg.sh probe, which makes the directory and submits)
#   cd /work/mech-ai-scratch/alloy/video-eeg-ensembling
#   sbatch [--dependency=afterok:<f16 cache job>] grader/sbatch_probe.sh
#   (without the f16s224 cache the sparse16 input is decoded on the fly with the identical rule)
#   then on CPU:  python grader/probe_analysis.py
#   DRY_RUN=1 bash grader/sbatch_probe.sh   # CPU, 10 clips, into output/ttg_tmp/probe_dryrun
set -u
case "${1:-}" in -h|--help) sed -n '2,/^set -u/p' "$0" | grep -v -e '^#SBATCH' -e '^set -u' | sed 's/^# \{0,1\}//'; exit 0 ;; esac
export EEG_ROOT=${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}   # data, caches, base trainers, outputs
EEG_ROOT=$(CDPATH= cd -- "$EEG_ROOT" 2>/dev/null && pwd -P) \
  || { echo "EEG_ROOT is not a directory"; exit 1; }   # physical path, as the Python side uses
# GRADER_DIR (the grader checkout): this script's own checkout on a direct `bash` call. Under
# sbatch "$0" is a spooled copy, so then $GRADER_DIR, else the checkout sbatch was run from
# ($SLURM_SUBMIT_DIR/grader, itself or its parent), else the default location.
_is_grader(){ [ -f "$1/train_grader.py" ] && [ -f "$1/eeg/train_eeg_det.py" ] && [ -f "$1/dhlib.py" ]; }
_submit_grader(){ local c; for c in "${SLURM_SUBMIT_DIR:-/nonexistent}"/{grader,.,..}; do
                    _is_grader "$c" && { echo "$c"; return 0; }; done; return 1; }
_here=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P)
if [ -n "$_here" ] && _is_grader "$_here"; then GRADER_DIR=$_here       # direct bash call
elif [ -n "${GRADER_DIR:-}" ]; then :                                  # exported by the helpers
elif _s=$(_submit_grader); then GRADER_DIR=$_s                        # sbatch from a checkout
else GRADER_DIR=/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader; fi
_g=$(CDPATH= cd -- "$GRADER_DIR" 2>/dev/null && pwd -P) && _is_grader "$_g" \
  || { echo "no grader checkout (train_grader.py, eeg/train_eeg_det.py, dhlib.py) in GRADER_DIR=$GRADER_DIR"; exit 1; }
GRADER_DIR=$_g
cd "$EEG_ROOT" || exit 1                         # the extractor's relative paths are EEG_ROOT's
ENV=/work/mech-ai-scratch/alloy/.conda/envs/eeg
export PATH="$ENV/bin:$PATH"
export TORCH_HOME=/work/mech-ai-scratch/alloy/.cache/torch
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
OUT=$EEG_ROOT/output/ttg_probe
W=$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python "$GRADER_DIR/probe_extract.py" --dry_run --limit 10 --workers 2 --verify_seek 2 \
       --out_dir "$EEG_ROOT/output/ttg_tmp/probe_dryrun"
fi
if [ -f "$OUT/features.npz" ]; then
  echo "[$(date +%F_%T)] SKIP: $OUT/features.npz exists"; exit 0
fi
echo "[$(date +%F_%T)] START probe extraction (job ${SLURM_JOB_ID:-local}, $(hostname), $W workers)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
# SIGTERM (preemption / scancel) and USR1 (time limit) stop the extractor; finished shards
# survive. After USR1 this job requeues itself; after a preemption SLURM requeues it.
child=""
usr1=0
trap 'echo "[$(date +%F_%T)] SIGTERM: stopping extractor"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM
trap 'echo "[$(date +%F_%T)] SIGUSR1 (time limit in 300 s): stopping extractor, then requeue"; usr1=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' USR1
python "$GRADER_DIR/probe_extract.py" --out_dir "$OUT" --workers "$W" --batch_clips 4 --verify_seek 20 \
       --retry_failed &
child=$!
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break      # still alive (signal interrupted wait): wait again
done
echo "[$(date +%F_%T)] END rc=$rc"
if [ "$usr1" -eq 1 ] && [ ! -f "$OUT/features.npz" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[$(date +%F_%T)] requeueing job $SLURM_JOB_ID (finished shards are kept)"
    scontrol requeue "$SLURM_JOB_ID" || echo "scontrol requeue failed: resubmit $GRADER_DIR/sbatch_probe.sh"
  fi
  exit 0
fi
exit $rc
