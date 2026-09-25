#!/bin/bash
#SBATCH --job-name=step0-analysis
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/step0an_%A_%a.out
#
# ==========================================================================================
#  Step 0 analysis (CPU only): grader/step0_vjepa_analysis.py as SLURM jobs.
# ==========================================================================================
#
# PURPOSE
#   The pre-registered evaluation (grader/step0_vjepa_prereg.md, section 8) runs as scavenger CPU
#   jobs: one prepare job, one array task per (contrast, arm) (18 tasks, 0..17; see
#   `python grader/step0_vjepa_analysis.py --list_jobs`), then one assemble job. No GPU.
#   The script pins OMP/OPENBLAS threads to 4 and the OpenBLAS kernel to Haswell itself, so the
#   fits are bit-identical on every node type. Each outer fold is cached atomically, so a
#   preempted (requeued) task resumes at the next unfinished fold.
#
# USAGE (from a login node; the eeg env python; outputs under $EEG_ROOT/output/ttg_vjepa/)
#   mkdir -p /work/mech-ai-scratch/alloy/EEG/logs/ttg
#   cd /work/mech-ai-scratch/alloy/video-eeg-ensembling
#   P=$(sbatch --parsable grader/sbatch_step0_analysis.sh prepare)
#   J=$(sbatch --parsable --dependency=afterok:$P --array=0-17 grader/sbatch_step0_analysis.sh job)
#   sbatch --dependency=afterok:$J grader/sbatch_step0_analysis.sh assemble
#   Extra arguments after MODE are passed to the script (tests: --vjepa <npz> --out
#   $EEG_ROOT/output/ttg_tmp/vjepa_<name> ...; the real output directory refuses test inputs).
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=3 bash grader/sbatch_step0_analysis.sh job   # print the command
set -u
case "${1:-}" in -h|--help|"") sed -n '2,/^set -u/p' "$0" | grep -v -e '^#SBATCH' -e '^set -u' | sed 's/^# \{0,1\}//'; exit 0 ;; esac
MODE=$1; shift
export EEG_ROOT=${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}
EEG_ROOT=$(CDPATH= cd -- "$EEG_ROOT" 2>/dev/null && pwd -P) || { echo "EEG_ROOT is not a directory"; exit 1; }
# GRADER_DIR: this script's own checkout on a direct `bash` call; under sbatch "$0" is a spooled
# copy, so then $GRADER_DIR, else the checkout sbatch was run from, else the default location.
_is_grader(){ [ -f "$1/step0_vjepa_analysis.py" ] && [ -f "$1/probe_analysis.py" ] && [ -f "$1/dhlib.py" ]; }
_submit_grader(){ local c; for c in "${SLURM_SUBMIT_DIR:-/nonexistent}"/{grader,.,..}; do
                    _is_grader "$c" && { echo "$c"; return 0; }; done; return 1; }
_here=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P)
if [ -n "$_here" ] && _is_grader "$_here"; then GRADER_DIR=$_here
elif [ -n "${GRADER_DIR:-}" ]; then :
elif _s=$(_submit_grader); then GRADER_DIR=$_s
else GRADER_DIR=/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader; fi
_g=$(CDPATH= cd -- "$GRADER_DIR" 2>/dev/null && pwd -P) && _is_grader "$_g" \
  || { echo "no grader checkout with step0_vjepa_analysis.py in GRADER_DIR=$GRADER_DIR"; exit 1; }
GRADER_DIR=$_g
PY=/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
case "$MODE" in
  prepare)  CMD=("$PY" "$GRADER_DIR/step0_vjepa_analysis.py" --stage prepare "$@") ;;
  job)      [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "job mode needs an array task id (--array=0-17)"; exit 1; }
            CMD=("$PY" "$GRADER_DIR/step0_vjepa_analysis.py" --stage jobs --job_index "$SLURM_ARRAY_TASK_ID" "$@") ;;
  assemble) CMD=("$PY" "$GRADER_DIR/step0_vjepa_analysis.py" --stage assemble "$@") ;;
  *) echo "MODE must be prepare | job | assemble, got $MODE"; exit 1 ;;
esac
if [ "${DRY_RUN:-0}" = "1" ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
echo "[$(date +%F_%T)] START step0 $MODE (job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:--}, $(hostname))"
child=""
usr1=0
trap 'echo "[$(date +%F_%T)] SIGTERM: stopping"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM
trap 'echo "[$(date +%F_%T)] SIGUSR1 (time limit in 300 s): stopping, then requeue"; usr1=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' USR1
"${CMD[@]}" &
child=$!
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break
done
echo "[$(date +%F_%T)] END rc=$rc"
if [ "$usr1" -eq 1 ] && [ "$MODE" = "job" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[$(date +%F_%T)] requeueing $SLURM_JOB_ID (finished folds are kept)"
  scontrol requeue "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null \
    || scontrol requeue "$SLURM_JOB_ID" || echo "scontrol requeue failed: resubmit task $SLURM_ARRAY_TASK_ID"
  exit 0
fi
exit $rc
