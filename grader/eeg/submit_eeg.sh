#!/bin/bash
# ==========================================================================================
#  Submission helper for the EGRG EEG-detector arrays (run on a login node, NOT via sbatch).
#  DRY-RUNS BY DEFAULT: nothing is created or submitted unless DRY_RUN=0.
# ==========================================================================================
#
# PURPOSE
#   Submits grader/eeg/sbatch_eeg_det.sh as one array per config table, with --array sized from
#   the table, after checking that the SLURM log directory exists (SLURM does not create it,
#   and a missing one makes every task fail at launch without a log) and that the segment
#   cache is present. In the default dry run it prints the sbatch command and, for every array
#   task, the exact trainer command the task would run (sbatch_eeg_det.sh with DRY_RUN=1).
#     aligned   grader/eeg/eeg_aligned.tsv   3 tasks  (aligned 5,279-clip split, seeds 1,2,3)
#     subject   grader/eeg/eeg_subject.tsv  10 tasks  (subject folds 0-4 x seeds 1,2)
#     all       both arrays
#     video     the video subject folds (grader/eeg/video_subject.tsv, 10 tasks) through the EXISTING
#               video driver grader/sbatch_grader.sh via grader/submit_ttg.sh stage1. CAUTION:
#               grader/submit_ttg.sh itself SUBMITS BY DEFAULT (DRY_RUN=0 there); this wrapper keeps
#               this script's dry-run default and passes DRY_RUN through explicitly.
#   The seed-3 tables (eeg_subject_s3.tsv, video_subject_s23.tsv) have no shortcut here: submit
#   them with sbatch --array sized from the table (see grader/README.md).
#   The drivers are taken from this script's checkout (GRADER_DIR = its parent directory) and
#   GRADER_DIR / EEG_ROOT are exported into the jobs; EEG_ROOT (env var, default
#   /work/mech-ai-scratch/alloy/EEG) holds the cache, the outputs and the logs.
#
# USAGE (any cwd)
#   bash grader/eeg/submit_eeg.sh aligned            # dry run (default): prints the plan
#   DRY_RUN=0 bash grader/eeg/submit_eeg.sh aligned  # really submit
#   DRY_RUN=0 bash grader/eeg/submit_eeg.sh all
#   bash grader/eeg/submit_eeg.sh video              # dry run of the video subject folds
#   DRY_RUN=0 bash grader/eeg/submit_eeg.sh video    # really submit them
#   bash grader/eeg/submit_eeg.sh --help
set -u
case "${1:-}" in -h|--help) sed -n '2,/^set -u/p' "$0" | grep -v -e '^set -u' | sed 's/^# \{0,1\}//'; exit 0 ;; esac
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
if [ -n "$_here" ] && _is_grader "${_here%/*}"; then GRADER_DIR=${_here%/*}       # direct bash call
elif [ -n "${GRADER_DIR:-}" ]; then :                                  # exported by the helpers
elif _s=$(_submit_grader); then GRADER_DIR=$_s                        # sbatch from a checkout
else GRADER_DIR=/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader; fi
_g=$(CDPATH= cd -- "$GRADER_DIR" 2>/dev/null && pwd -P) && _is_grader "$_g" \
  || { echo "no grader checkout (train_grader.py, eeg/train_eeg_det.py, dhlib.py) in GRADER_DIR=$GRADER_DIR"; exit 1; }
GRADER_DIR=$_g
export GRADER_DIR                                # read by the job scripts under sbatch
cd "$EEG_ROOT" || exit 1
LOGS=$EEG_ROOT/logs/ttg
CACHE=cache_bestcfg/seg_w6.0_s3.0_d8.npz
DRIVER=$GRADER_DIR/eeg/sbatch_eeg_det.sh
dry=${DRY_RUN:-1}
what=${1:-}

if [ "$what" = "video" ]; then
  VT=$GRADER_DIR/eeg/video_subject.tsv
  [ -f "$VT" ] || { echo "no table $VT"; exit 1; }
  N=$(grep -cvE '^[[:space:]]*(#|$)' "$VT")
  if [ "$dry" = "1" ]; then
    echo "DRY RUN (default): nothing is created or submitted; DRY_RUN=0 to submit"
    for ((i = 0; i < N; i++)); do
      DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash "$GRADER_DIR/sbatch_grader.sh" "$VT" | sed 's/^/    /'
    done
  fi
  exec env DRY_RUN="$dry" bash "$GRADER_DIR/submit_ttg.sh" stage1 "$VT"
fi

case "$what" in
  aligned) TABLES=("$GRADER_DIR/eeg/eeg_aligned.tsv") ;;
  subject) TABLES=("$GRADER_DIR/eeg/eeg_subject.tsv") ;;
  all)     TABLES=("$GRADER_DIR/eeg/eeg_aligned.tsv" "$GRADER_DIR/eeg/eeg_subject.tsv") ;;
  *) echo "usage: [DRY_RUN=0] bash grader/eeg/submit_eeg.sh {aligned|subject|all|video}"; exit 1 ;;
esac
[ "$dry" = "1" ] && echo "DRY RUN (default): nothing is created or submitted; DRY_RUN=0 to submit"

if [ ! -d "$LOGS" ]; then
  if [ "$dry" = "1" ]; then echo "would run: mkdir -p $LOGS"; else mkdir -p "$LOGS" || { echo "cannot create $LOGS"; exit 1; }; fi
fi
[ -f "$CACHE" ] || { echo "REFUSING: segment cache $CACHE is missing"; exit 1; }

for TABLE in "${TABLES[@]}"; do
  [ -f "$TABLE" ] || { echo "no table $TABLE"; exit 1; }
  N=$(grep -cvE '^[[:space:]]*(#|$)' "$TABLE")
  [ "$N" -ge 1 ] || { echo "empty table $TABLE"; exit 1; }
  # every line must parse before anything is submitted
  for ((i = 0; i < N; i++)); do
    DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash "$DRIVER" "$TABLE" > /dev/null \
      || { echo "REFUSING: line $i of $TABLE does not parse:"; DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash "$DRIVER" "$TABLE"; exit 1; }
  done
  cmd=(sbatch --parsable --array=0-$((N - 1)) "$DRIVER" "$TABLE")
  if [ "$dry" = "1" ]; then
    echo "would run: ${cmd[*]}"
    for ((i = 0; i < N; i++)); do
      DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash "$DRIVER" "$TABLE" | sed 's/^/    /'
    done
  else
    jid=$("${cmd[@]}") || { echo "submission of $TABLE failed"; exit 1; }
    echo "submitted $TABLE: array job $jid (tasks 0-$((N - 1)))"
  fi
done
echo "logs: $LOGS/eegdet_<jobid>_<task>.out and $LOGS/eegdet_<name>.log"
