#!/bin/bash
# ==========================================================================================
#  Submission helper for the TT-X3D SLURM jobs (run on a login node, NOT via sbatch).
# ==========================================================================================
#
# PURPOSE
#   The three job scripts write their SLURM logs to $EEG_ROOT/logs/ttg/, and SLURM does not
#   create that directory: without it every job fails at launch and leaves no log (sbatch
#   --test-only does not catch this). This helper makes the directory first, then submits
#   with the right dependencies:
#     cache   grader/sbatch_cache.sh (nova CPU job) unless cache_frames/f16s224/index.json exists
#     stage1  the cache job if needed, then the Stage-1 array (one task per line of
#             grader/stage1.tsv) with --dependency=afterok:<cache job>. sbatch_cache.sh exits 1
#             if the published index lists any unreadable clip, so afterok holds the array
#             instead of releasing 11 tasks that would all refuse to start. With an existing
#             cache, the unreadable list is checked here before submitting.
#     probe   grader/sbatch_probe.sh (Stage-2 feature extraction; needs no cache, it decodes
#             the sparse16 input itself when f16s224 is absent)
#   The job scripts are submitted from this script's directory (GRADER_DIR, the
#   video-eeg-ensembling/grader checkout it lives in) with GRADER_DIR and EEG_ROOT exported
#   into the job; they run the Python with cwd = EEG_ROOT (env var, default
#   /work/mech-ai-scratch/alloy/EEG), which holds the data, caches, outputs and logs.
#
# USAGE (any cwd; a relative table is resolved against the cwd, then against grader/)
#   bash grader/submit_ttg.sh stage1 [table]   # default table grader/stage1.tsv
#   bash grader/submit_ttg.sh cache | probe
#   DRY_RUN=1 bash grader/submit_ttg.sh stage1 # print the plan; creates and submits nothing
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
export GRADER_DIR                                # read by the job scripts under sbatch
TABLE=${2:-$GRADER_DIR/stage1.tsv}               # resolved before the cd below
case "$TABLE" in /*) ;; *) if [ -f "$TABLE" ]; then TABLE=$PWD/$TABLE
                           elif [ -f "$GRADER_DIR/$TABLE" ]; then TABLE=$GRADER_DIR/$TABLE
                           else TABLE=$PWD/$TABLE; fi ;; esac   # never left relative: the cd re-anchors it
cd "$EEG_ROOT" || exit 1
LOGS=$EEG_ROOT/logs/ttg
CACHE=${TTG_CACHE:-cache_frames/f16s224}     # override only for local tests
PY=/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python
what=${1:-}
dry=${DRY_RUN:-0}

run(){ if [ "$dry" = "1" ]; then echo "DRY_RUN: $*" >&2; echo "<jobid>"; else "$@"; fi; }

case "$what" in cache|stage1|probe) ;;
  *) echo "usage: bash grader/submit_ttg.sh {cache|stage1|probe} [table]"; exit 1 ;; esac

if [ "$dry" = "1" ]; then
  echo "DRY_RUN: mkdir -p $LOGS"
else
  mkdir -p "$LOGS" || { echo "cannot create $LOGS"; exit 1; }
fi

cache_dep=""
if [ "$what" = "cache" ] || [ "$what" = "stage1" ]; then
  if [ -f "$CACHE/index.json" ]; then
    n_bad=$("$PY" -c "import json;print(len(json.load(open('$CACHE/index.json')).get('unreadable',[])))")
    echo "$CACHE/index.json exists ($n_bad unreadable clips); no cache job needed"
    if [ "$n_bad" != "0" ]; then
      echo "REFUSING to submit: train_grader.py rejects any unreadable train/val clip"; exit 1
    fi
  else
    jid=$(run sbatch --parsable "$GRADER_DIR/sbatch_cache.sh") || { echo "cache submission failed"; exit 1; }
    echo "cache job: $jid"
    cache_dep="--dependency=afterok:${jid%%;*}"
  fi
fi

if [ "$what" = "stage1" ]; then
  [ -f "$TABLE" ] || { echo "no table $TABLE"; exit 1; }
  N=$(grep -cvE '^[[:space:]]*(#|$)' "$TABLE")
  [ "$N" -ge 1 ] || { echo "empty table $TABLE"; exit 1; }
  jid=$(run sbatch --parsable $cache_dep --array=0-$((N - 1)) "$GRADER_DIR/sbatch_grader.sh" "$TABLE") \
    || { echo "grader submission failed"; exit 1; }
  echo "Stage-1 array: $jid (tasks 0-$((N - 1)) from $TABLE) ${cache_dep:+[$cache_dep]}"
fi

if [ "$what" = "probe" ]; then
  jid=$(run sbatch --parsable "$GRADER_DIR/sbatch_probe.sh") || { echo "probe submission failed"; exit 1; }
  echo "probe job: $jid"
fi
echo "logs: $LOGS"
