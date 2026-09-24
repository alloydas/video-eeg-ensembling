#!/bin/bash
#SBATCH --job-name=ttg-eegdet
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/eegdet_%A_%a.out
#
# ==========================================================================================
#  SLURM array driver for grader/eeg/train_eeg_det.py (EGRG EEG detector: TCN, binary, fixed
#  last epoch, per-epoch clip + window dumps).
# ==========================================================================================
#
# PURPOSE
#   One array task = one line of a whitespace-separated config table:
#       name  split  universe  fold  seed  output  [extra trainer flags...]
#   split is session|subject; universe is video|eeg for session rows and '-' for subject rows
#   (the trainer refuses --session_universe video with --split subject); fold is ignored for
#   session rows; output is an ABSOLUTE dir under /work/mech-ai-scratch/alloy/EEG/output/ttg_eeg*.
#   Blank lines and lines starting with # are ignored; the i-th remaining line is task i.
#     grader/eeg/eeg_aligned.tsv     aligned 5,279-clip split, seeds 1,2,3  (3 lines -> --array=0-2)
#     grader/eeg/eeg_subject.tsv     subject folds 0-4 x seeds 1,2         (10 lines -> --array=0-9)
#     grader/eeg/eeg_subject_s3.tsv  subject folds 0-4, seed 3              (5 lines -> --array=0-4)
#   There is NO default --array: the table decides it. A job submitted without --array is refused
#   (it would silently run line 0 only), and an array that covers fewer lines than the table
#   prints a warning into every task's log (deliberate subsets stay possible).
#   Fixed flags for every row (train_pooled_eeg.py defaults, the v3_bestcfg cache and pooling):
#     --arch tcn --epochs 30 --batch_size 256 --lr 1e-3 --agg logmean --split_seed 49
#     --cache cache_bestcfg/seg_w6.0_s3.0_d8.npz --require_cuda
#
# WHERE THINGS ARE
#   code     GRADER_DIR = video-eeg-ensembling/grader (this script lives in grader/eeg). Under
#            sbatch "$0" is a spooled copy, so the script's own checkout is used only on a
#            direct `bash` call; otherwise $GRADER_DIR (exported by the submit helpers), else
#            the checkout sbatch was run from ($SLURM_SUBMIT_DIR/grader, or $SLURM_SUBMIT_DIR
#            if that is grader/ or grader/eeg), else
#            /work/mech-ai-scratch/alloy/video-eeg-ensembling/grader.
#   data     EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG): the trainer runs with
#            cwd = EEG_ROOT (segment cache, video index, the imported train_pooled_eeg, the
#            parent's check_eeg_align.py), outputs go to $EEG_ROOT/output/ttg_eeg*, logs to
#            $EEG_ROOT/logs/ttg.
#   table    a relative table path is resolved against the directory sbatch / bash was run
#            from, then against grader/eeg/, then against grader/; a table found in none of
#            them is refused (it is never looked up under EEG_ROOT).
#
# USAGE (grader/eeg/submit_eeg.sh does all of this, and dry-runs unless DRY_RUN=0)
#   cd /work/mech-ai-scratch/alloy/video-eeg-ensembling
#   N=$(grep -cvE '^[[:space:]]*(#|$)' grader/eeg/eeg_aligned.tsv)
#   sbatch --array=0-$((N-1)) grader/eeg/sbatch_eeg_det.sh grader/eeg/eeg_aligned.tsv
#   # what task i would run, without SLURM, locks or a GPU:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash grader/eeg/sbatch_eeg_det.sh grader/eeg/eeg_aligned.tsv
#   (EEGDET_LOG_DIR / EEGDET_TEST_FLAGS override the log dir / --require_cuda for local CPU
#    tests only; later flags win, so e.g. EEGDET_TEST_FLAGS="--allow_cpu --epochs 2 --limit 800".)
#
# PARTITION FACTS (scontrol show partition scavenger; same design as grader/sbatch_grader.sh)
#   * PreemptMode=REQUEUE, GraceTime=0, KillWait=30 s: on preemption every process gets
#     SIGTERM, then SIGKILL 30 s later, and SLURM requeues the job. The trainer handles SIGTERM
#     itself (checkpoint at the next step boundary -- milliseconds on a GPU -- then exit 3);
#     the trap below also forwards it. The requeued task finds last.pt and resumes mid-epoch
#     BIT-identically (verify_eeg_det.py resume). last.pt is also written every epoch end.
#   * --time 4:00:00 (a run is ~21 A100-min per the v3_eegalign logs); --signal=B:USR1@300
#     warns this shell 5 min before the limit: the trainer checkpoints and the task is requeued
#     with `scontrol requeue` (a plain SIGTERM is NOT requeued by this script, so scancel works).
#   * --constraint a100|h200|l40s: the a40 nodes carry no feature tag and the rtx_pro_6000
#     (sm_120) nodes cannot run torch 2.4.1+cu121; the trainer also exits 4 on such a GPU.
#
# SAFETY
#   * a task whose output already holds results.json is skipped;
#   * an atomic mkdir lock stops two runners training into one output dir (a requeued task,
#     same SLURM_JOB_ID, reclaims its own lock; a dead owner's lock is stolen); the trainer
#     additionally holds an flock on <output>/.run.lock;
#   * preflight: the cache exists, the env imports, and for aligned rows the parent repo's
#     $EEG_ROOT/check_eeg_align.py (read-only, 12 hard checks) proves the aligned split before the GPU is used;
#   * nothing is ever deleted: last.pt / val_clip_ep*.npz / history.json all stay.
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
if [ -n "$_here" ] && _is_grader "${_here%/*}"; then GRADER_DIR=${_here%/*}       # direct bash call
elif [ -n "${GRADER_DIR:-}" ]; then :                                  # exported by the helpers
elif _s=$(_submit_grader); then GRADER_DIR=$_s                        # sbatch from a checkout
else GRADER_DIR=/work/mech-ai-scratch/alloy/video-eeg-ensembling/grader; fi
_g=$(CDPATH= cd -- "$GRADER_DIR" 2>/dev/null && pwd -P) && _is_grader "$_g" \
  || { echo "no grader checkout (train_grader.py, eeg/train_eeg_det.py, dhlib.py) in GRADER_DIR=$GRADER_DIR"; exit 1; }
GRADER_DIR=$_g
TABLE=${1:-$GRADER_DIR/eeg/eeg_aligned.tsv}     # resolved before the cd below
case "$TABLE" in /*) ;; *) if [ -f "$TABLE" ]; then TABLE=$PWD/$TABLE
                           elif [ -f "$GRADER_DIR/eeg/$TABLE" ]; then TABLE=$GRADER_DIR/eeg/$TABLE
                           elif [ -f "$GRADER_DIR/$TABLE" ]; then TABLE=$GRADER_DIR/$TABLE
                           else TABLE=$PWD/$TABLE; fi ;; esac   # never left relative: the cd re-anchors it
cd "$EEG_ROOT" || exit 1                         # the trainer's relative paths are EEG_ROOT's
ENV=/work/mech-ai-scratch/alloy/.conda/envs/eeg
export PATH="$ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
L=${EEGDET_LOG_DIR:-$EEG_ROOT/logs/ttg}      # override only for local tests
CACHE=cache_bestcfg/seg_w6.0_s3.0_d8.npz
log(){ echo "[$(date +%F_%T)] $*"; }

[ -f "$TABLE" ] || { log "no config table $TABLE"; exit 1; }
NROWS=$(grep -cvE '^[[:space:]]*(#|$)' "$TABLE")
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  log "REFUSING: submitted without --array; $TABLE has $NROWS lines -> sbatch --array=0-$((NROWS - 1)) (or use $GRADER_DIR/eeg/submit_eeg.sh)"
  exit 1
fi
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT}" -lt "$NROWS" ]; then
  log "WARNING: this array has ${SLURM_ARRAY_TASK_COUNT} tasks but $TABLE has $NROWS lines: lines outside --array will NOT run"
fi
i=${SLURM_ARRAY_TASK_ID:-0}
line=$(grep -vE '^[[:space:]]*(#|$)' "$TABLE" | sed -n "$((i + 1))p")
[ -n "$line" ] || { log "no line $i in $TABLE"; exit 1; }
read -r name split universe fold seed out extra <<< "$line"
extra=${extra:-}
[ "$extra" = "-" ] && extra=""

case "$split" in
  session) case "$universe" in video|eeg) SPLIT=(--split session --session_universe "$universe") ;;
             *) log "session row needs universe video|eeg, got '$universe' (line $i)"; exit 1 ;; esac ;;
  subject) [ "$universe" = "-" ] || { log "subject row must have universe '-' (got '$universe', line $i)"; exit 1; }
           [[ "$fold" =~ ^[0-4]$ ]] || { log "bad fold '$fold' (line $i)"; exit 1; }
           SPLIT=(--split subject --fold "$fold" --n_folds 5) ;;
  *) log "bad split '$split' (session|subject) in line $i"; exit 1 ;;
esac
[[ "$seed" =~ ^[0-9]+$ ]] || { log "bad seed '$seed' (line $i)"; exit 1; }
case "$out" in /*) out_real=$(realpath -m -- "$out") ;; *) out_real="" ;; esac   # symlinks resolved
case "$out_real" in "$EEG_ROOT"/output/ttg_eeg*) ;; *) log "output must be absolute under $EEG_ROOT/output/ttg_eeg*: $out"; exit 1 ;; esac

REQ=(--require_cuda)
# LOCAL CPU TESTS ONLY: EEGDET_TEST_FLAGS replaces --require_cuda (e.g. "--allow_cpu --limit 800
# --epochs 2 --threads 4"); never set it in a SLURM submission.
[ -n "${EEGDET_TEST_FLAGS:-}" ] && read -r -a REQ <<< "$EEGDET_TEST_FLAGS"
CMD=(python "$GRADER_DIR/eeg/train_eeg_det.py" "${SPLIT[@]}" --seed "$seed" --split_seed 49 --arch tcn
     --epochs 30 --batch_size 256 --lr 1e-3 --agg logmean --cache "$CACHE" "${REQ[@]}"
     --output "$out" $extra)

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "task $i ($name): ${CMD[*]}"
  exit 0
fi

if [ -f "$out/results.json" ]; then
  log "SKIP $name (already has results.json)"; exit 0
fi
mkdir -p "$L" "$(dirname "$out")"

# ---- atomic claim (reclaim after requeue; steal from a dead job) --------------------------
LOCKDIR="$(dirname "$out")/.locks"
mkdir -p "$LOCKDIR"
lock="$LOCKDIR/$name"
me="${SLURM_JOB_ID:-local-$$}"
if mkdir "$lock" 2>/dev/null; then
  echo "$me $(hostname) $(date +%F_%T)" > "$lock/owner"
else
  owner=$(cut -d' ' -f1 "$lock/owner" 2>/dev/null || echo "")
  if [ "$owner" = "$me" ]; then
    log "RECLAIM $name after requeue"
  elif [ -n "$owner" ] && squeue -j "$owner" -h -o %T 2>/dev/null | grep -q .; then
    log "SKIP $name (held by live job $owner)"; exit 0
  elif [[ "$owner" == local-* ]] && [ "$(cut -d' ' -f2 "$lock/owner")" = "$(hostname)" ] \
       && kill -0 "${owner#local-}" 2>/dev/null; then
    log "SKIP $name (held by live local process ${owner#local-})"; exit 0
  elif [ -f "$out/results.json" ]; then
    log "SKIP $name (finished by $owner)"; exit 0
  else
    log "STEAL $name from dead job ${owner:-unknown}"
    echo "$me $(hostname) $(date +%F_%T) stole-from:${owner:-unknown}" > "$lock/owner"
  fi
fi
trap '[ -f "$out/results.json" ] || rm -rf "$lock"' EXIT

# ---- preflight --------------------------------------------------------------------------
[ -f "$CACHE" ] || { log "PREFLIGHT FAILED: missing $CACHE"; exit 1; }
python -c "import torch, sklearn, numpy" || { log "PREFLIGHT FAILED: env imports"; exit 1; }
if [ "$split" = "session" ] && [ "$universe" = "video" ]; then
  python "$EEG_ROOT/check_eeg_align.py" --cache "$CACHE" --split_seed 49 > "$L/eegdet_${name}.preflight.log" 2>&1 \
    || { log "PREFLIGHT FAILED: check_eeg_align.py (see $L/eegdet_${name}.preflight.log)"; exit 1; }
fi

log "START $name (job ${SLURM_JOB_ID:-local} task $i on $(hostname)); log $L/eegdet_$name.log"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
log "CMD ${CMD[*]}"

# ---- run, forwarding preemption / time-limit signals to the trainer -----------------------
child=""
usr1=0
fwd(){ [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; }
trap 'log "SIGTERM (preemption or scancel): forwarding to trainer"; fwd' TERM
trap 'log "SIGUSR1 (time limit in 300 s): checkpoint, then requeue"; usr1=1; fwd' USR1
"${CMD[@]}" >> "$L/eegdet_$name.log" 2>&1 &
child=$!
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break      # still alive (a trap interrupted wait): wait again
done

if [ "$rc" -eq 0 ] && [ -f "$out/results.json" ]; then
  log "DONE $name"; exit 0
elif [ "$rc" -eq 3 ]; then
  if [ "$usr1" -eq 1 ]; then
    log "CHECKPOINTED $name before the time limit; requeueing"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then        # only ever requeue THIS array task
      scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
        || log "scontrol requeue failed: resubmit task $i"
    else
      log "not an array task: resubmit by hand"
    fi
  else
    log "CHECKPOINTED $name after SIGTERM; SLURM requeues preempted jobs itself"
  fi
  exit 0
elif [ "$rc" -eq 4 ]; then
  log "FAILED $name: unsupported GPU on $(hostname) (exit 4); check --constraint"; exit 1
else
  log "FAILED $name (exit $rc); tail of $L/eegdet_$name.log:"; tail -5 "$L/eegdet_$name.log"; exit 1
fi
