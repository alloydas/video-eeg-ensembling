#!/bin/bash
#SBATCH --job-name=ttg-vjepa
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
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/vjepa_%j.out
#
# ==========================================================================================
#  Step 0: frozen V-JEPA 2 ViT-L features (sparse16 / dense / dense_shuffled) for every clip.
# ==========================================================================================
#
# PURPOSE
#   Runs grader/step0_vjepa_extract.py (pre-registration: grader/step0_vjepa_prereg.md) with the
#   V-JEPA 2 venv python (/work/mech-ai-scratch/alloy/.venvs/vjepa2, never the shared eeg env).
#   Decoding (sparse16 linspace + 8 frame-exact seeks x 31 native frames per clip, at 256) runs in
#   14 DataLoader workers; the GPU runs 17 views per clip, bf16 autocast. Output:
#   $EEG_ROOT/output/ttg_vjepa/ (shards/, budget/, budget.json; features.npz after the merge).
#   The extractor is $GRADER_DIR/step0_vjepa_extract.py (video-eeg-ensembling/grader; under
#   sbatch "$0" is a spooled copy, so $GRADER_DIR, else the checkout sbatch was run from, else
#   the fixed default is used), run with cwd = EEG_ROOT (env var,
#   default /work/mech-ai-scratch/alloy/EEG), HF_HOME=/work/mech-ai-scratch/alloy/hf_cache and
#   HF_HUB_OFFLINE=1 (the pinned snapshot was downloaded on the login node).
#
# MODES (first argument)
#   (none) | START STOP   GPU extraction of shards [START, STOP) (default: all). Shards 0..23 are
#                         pass 1 (seizure clips), 24..48 pass 2 (non-seizure); several jobs may
#                         split the range, but ranges must not overlap. GPU jobs never merge.
#   prepare               CPU: the item list, --verify_seek 20 at 256, run.json. Run it once,
#                         before the GPU jobs, under srun (below); a GPU job that finds no
#                         run.json prepares by itself (2-4 min of GPU idle).
#   merge [--allow_pass2_incomplete]   CPU: assemble and re-check features.npz.
#   status                prints shard and budget status (login node is fine).
#   DRY_RUN=1 (any mode ignored): the pre-registered CPU dry run on the first 3 sorted clips,
#                         into output/ttg_tmp/vjepa_dryrun (run it under srun -c 8).
#
# BUDGET, PREEMPTION AND TIME LIMIT
#   The cap is 8 GPU-hours over every job of the run (section 8). This driver exports
#   VJ_SEGMENT_T0 (its start time) and writes $OUT/budget/job_<id>_r<restart>.txt, so the
#   extractor's budget.json and its sacct reconciliation see every segment, including one killed
#   before Python started. When the cap stops a job the extractor exits 5 and the job is NOT
#   requeued. Scavenger requeues preempted jobs (PreemptMode=REQUEUE, KillWait 30 s); the
#   extractor records its budget on SIGTERM and exits 3 (or 1, when a DataLoader worker was
#   signalled first; after a stop signal this driver treats any exit code as a stop), finished
#   512-clip shards are kept and a requeued job resumes after them. --signal=B:USR1@300 warns this shell 5 min before the time
#   limit: it stops the extractor and requeues the job with `scontrol requeue` (a plain SIGTERM,
#   e.g. scancel, is not requeued here). --retry_failed makes every restart re-extract rows that
#   failed in finished shards.
#
# USAGE (from the video-eeg-ensembling checkout; SLURM does not create the log directory)
#   mkdir -p /work/mech-ai-scratch/alloy/EEG/logs/ttg
#   srun -A mech-ai-scavenger -q scavenger -p scavenger -c 8 --mem=32G -t 01:00:00 \
#        bash grader/sbatch_vjepa.sh prepare
#   sbatch grader/sbatch_vjepa.sh                      # one GPU, all shards
#   sbatch grader/sbatch_vjepa.sh 0 12; sbatch grader/sbatch_vjepa.sh 12 24   # or split pass 1 ...
#   sbatch grader/sbatch_vjepa.sh 24 37; sbatch grader/sbatch_vjepa.sh 37 49  # ... then pass 2
#   srun -A mech-ai-scavenger -q scavenger -p scavenger -c 4 --mem=48G -t 01:00:00 \
#        bash grader/sbatch_vjepa.sh merge
#   bash grader/sbatch_vjepa.sh status
#   DRY_RUN=1 srun -A mech-ai-scavenger -q scavenger -p scavenger -c 8 --mem=32G -t 01:00:00 \
#        bash grader/sbatch_vjepa.sh
set -u
case "${1:-}" in -h|--help) sed -n '2,/^set -u/p' "$0" | grep -v -e '^#SBATCH' -e '^set -u' | sed 's/^# \{0,1\}//'; exit 0 ;; esac
T0=$(date +%s)
export EEG_ROOT=${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}   # data, caches, base trainers, outputs
EEG_ROOT=$(CDPATH= cd -- "$EEG_ROOT" 2>/dev/null && pwd -P) \
  || { echo "EEG_ROOT is not a directory"; exit 1; }   # physical path, as the Python side uses
# GRADER_DIR (the grader checkout): this script's own checkout on a direct `bash` call. Under
# sbatch "$0" is a spooled copy, so then $GRADER_DIR, else the checkout sbatch was run from
# ($SLURM_SUBMIT_DIR/grader, itself or its parent), else the default location.
_is_grader(){ [ -f "$1/train_grader.py" ] && [ -f "$1/eeg/train_eeg_det.py" ] && [ -f "$1/dhlib.py" ] \
                && [ -f "$1/step0_vjepa_extract.py" ]; }
_submit_grader(){ local c; for c in "${SLURM_SUBMIT_DIR:-/nonexistent}"/{grader,.,..}; do
                    _is_grader "$c" && { echo "$c"; return 0; }; done; return 1; }
_here=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P)
if [ -n "$_here" ] && _is_grader "$_here"; then GRADER_DIR=$_here       # direct bash call
elif [ -n "${GRADER_DIR:-}" ]; then :                                  # exported by a helper
elif _s=$(_submit_grader); then GRADER_DIR=$_s                        # sbatch from a checkout
else GRADER_DIR=/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader; fi
_g=$(CDPATH= cd -- "$GRADER_DIR" 2>/dev/null && pwd -P) && _is_grader "$_g" \
  || { echo "no grader checkout (train_grader.py, eeg/train_eeg_det.py, dhlib.py, step0_vjepa_extract.py) in GRADER_DIR=$GRADER_DIR"; exit 1; }
GRADER_DIR=$_g
cd "$EEG_ROOT" || exit 1                         # the extractor's relative paths are EEG_ROOT's
VENV=/work/mech-ai-scratch/alloy/.venvs/vjepa2
PY=$VENV/bin/python
export HF_HOME=/work/mech-ai-scratch/alloy/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/work/mech-ai-scratch/alloy/.cache/torch
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export VJ_SEGMENT_T0=$T0
OUT=$EEG_ROOT/output/ttg_vjepa
SCRIPT=$GRADER_DIR/step0_vjepa_extract.py
log(){ echo "[$(date +%F_%T)] $*"; }

"$PY" - <<'PY' || { log "PREFLIGHT FAILED: the venv must give transformers 4.53.0 (VJEPA2Model) over torch 2.4.1+cu121"; exit 1; }
import sys, torch, transformers
from transformers import VJEPA2Model  # noqa: F401
ok = transformers.__version__ == "4.53.0" and torch.__version__ == "2.4.1+cu121"
print(f"venv {sys.prefix}: torch {torch.__version__}, transformers {transformers.__version__}")
sys.exit(0 if ok else 1)
PY

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec "$PY" "$SCRIPT" --dry_run --limit 3 --verify_seek 3 --workers 2 --threads "${SLURM_CPUS_PER_TASK:-4}" \
       --out_dir "$EEG_ROOT/output/ttg_tmp/vjepa_dryrun"
fi
case "${1:-}" in
  prepare) exec "$PY" "$SCRIPT" --prepare --verify_seek 20 ;;
  merge)   shift; exec "$PY" "$SCRIPT" --merge "$@" ;;
  status)  exec "$PY" "$SCRIPT" --status ;;
esac
RANGE=()
if [ $# -ge 2 ]; then
  case "$1$2" in *[!0-9]*) log "usage: sbatch grader/sbatch_vjepa.sh [START STOP | prepare | merge | status]"; exit 1 ;; esac
  RANGE=(--shard_start "$1" --shard_stop "$2")
elif [ $# -eq 1 ]; then
  log "usage: sbatch grader/sbatch_vjepa.sh [START STOP | prepare | merge | status]"; exit 1
fi
if [ -f "$OUT/features.npz" ]; then
  log "SKIP: $OUT/features.npz exists"; exit 0
fi
if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
  # never record a non-GPU allocation (e.g. an interactive CPU job) in the GPU budget
  log "REFUSING: the extraction runs only inside a SLURM GPU job (sbatch); use DRY_RUN=1 for a CPU test"; exit 1
fi
mkdir -p "$OUT/budget" || exit 1
# one file per job start (appends to a shared file are not atomic across NFS clients)
echo "${SLURM_JOB_ID} ${SLURM_RESTART_COUNT:-0} $T0 $(hostname) ${RANGE[*]:-all}" \
  > "$OUT/budget/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}.txt"
W=$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))
log "START V-JEPA 2 extraction (job ${SLURM_JOB_ID:-local} restart ${SLURM_RESTART_COUNT:-0}, $(hostname), $W workers, shards ${RANGE[*]:-all})"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
# SIGTERM (preemption / scancel) and USR1 (time limit) stop the extractor, which records its
# budget first; finished shards survive. After USR1 this job requeues itself; after a
# preemption SLURM requeues it.
child=""
usr1=0
term=0
trap 'log "SIGTERM: stopping extractor"; term=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM
trap 'log "SIGUSR1 (time limit in 300 s): stopping extractor, then requeue"; usr1=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' USR1
"$PY" "$SCRIPT" --out_dir "$OUT" --workers "$W" --batch_clips 4 --chunk_views 68 --retry_failed "${RANGE[@]}" &
child=$!
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break      # still alive (signal interrupted wait): wait again
done
log "END rc=$rc"
case "$rc" in
  0) log "DONE shards ${RANGE[*]:-all}; run the merge (CPU) when every range is done"; exit 0 ;;
  5) log "BUDGET CAP reached (8 GPU-h, section 8): not requeueing; see $OUT/budget.json"; exit 0 ;;
  4) log "FAILED: unsupported GPU on $(hostname) (exit 4); check --constraint"; exit 1 ;;
  *) if [ "$rc" -ne 3 ] && [ "$usr1" -eq 0 ] && [ "$term" -eq 0 ]; then
       log "FAILED (exit $rc)"; exit 1
     fi
     # exit 3 is the extractor's own stop. On preemption SLURM also signals the DataLoader
     # workers, and the extractor can then die through PyTorch's worker-died error (exit 1)
     # before its SIGTERM handler runs; after a stop signal any exit code is a stop.
     [ "$rc" -ne 3 ] && log "extractor exit $rc after a stop signal (a DataLoader worker was probably signalled first): treated as a stop"
     if [ "$usr1" -eq 1 ]; then
       if [ -n "${SLURM_JOB_ID:-}" ]; then
         log "requeueing job $SLURM_JOB_ID (finished shards are kept)"
         scontrol requeue "$SLURM_JOB_ID" || log "scontrol requeue failed: resubmit $GRADER_DIR/sbatch_vjepa.sh ${RANGE[*]:-}"
       fi
     else
       log "stopped by SIGTERM; SLURM requeues preempted jobs itself"
     fi
     exit 0 ;;
esac
