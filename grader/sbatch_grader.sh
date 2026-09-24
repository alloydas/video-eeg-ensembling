#!/bin/bash
#SBATCH --job-name=ttg-grader
#SBATCH --partition=scavenger
#SBATCH --account=mech-ai-scavenger
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --array=0-10
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/grader_%A_%a.out
#
# ==========================================================================================
#  SLURM array driver for grader/train_grader.py (TT-X3D Stage 1 and later stages).
# ==========================================================================================
#
# PURPOSE
#   One array task = one line of a whitespace-separated config table:
#       name  arch  fix  heads  seed  split  fold  output  [extra trainer flags...]
#   fix is `fixed` (--fix_x3d) or `bugged` (historical head softmax); output is an
#   ABSOLUTE dir under /work/mech-ai-scratch/alloy/EEG/output/ttg_*. Blank lines and lines
#   starting with # are ignored; the i-th remaining line is array task i.
#   grader/stage1.tsv is the Stage-1 table (11 lines -> --array=0-10).
#
# WHERE THINGS ARE
#   code     GRADER_DIR = video-eeg-ensembling/grader. Under sbatch "$0" is a spooled copy,
#            so this script uses its own directory only when that directory is a grader
#            checkout (a direct `bash` call); otherwise $GRADER_DIR (exported by the submit
#            helpers), else the checkout sbatch was run from ($SLURM_SUBMIT_DIR/grader, or
#            $SLURM_SUBMIT_DIR if that is grader/ or grader/eeg), else
#            /work/mech-ai-scratch/alloy/video-eeg-ensembling/grader.
#   data     EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG): the trainer runs
#            with cwd = EEG_ROOT (caches, data/ keys, the imported train_pooled), outputs go
#            to $EEG_ROOT/output/ttg_*, logs to $EEG_ROOT/logs/ttg.
#   table    a relative table path is resolved against the directory sbatch / bash was run
#            from, then against GRADER_DIR (so `stage1b.tsv` or `eeg/video_subject.tsv` work);
#            a table found in neither is refused (it is never looked up under EEG_ROOT).
#
# USAGE (from anywhere; SLURM does not create the log directory, so make it first --
#        grader/submit_ttg.sh does that and the dependency wiring for you)
#   mkdir -p /work/mech-ai-scratch/alloy/EEG/logs/ttg
#   cd /work/mech-ai-scratch/alloy/video-eeg-ensembling
#   N=$(grep -cvE '^[[:space:]]*(#|$)' grader/stage1.tsv)
#   sbatch --array=0-$((N-1)) grader/sbatch_grader.sh grader/stage1.tsv
#   # inspect what task i would run, without SLURM, locks or a GPU:
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=0 bash grader/sbatch_grader.sh grader/stage1.tsv
#   (TTG_LOG_DIR / TTG_CACHE override the log dir and cache for local CPU tests only.)
#
# PARTITION FACTS THIS SCRIPT IS BUILT AROUND (scontrol show partition scavenger)
#   * PreemptMode=REQUEUE, GraceTime=0, KillWait=30 s: on preemption every process of the
#     job gets SIGTERM, then SIGKILL 30 s later, and SLURM requeues the job itself. The
#     trap below forwards SIGTERM to the trainer, which checkpoints at the next step
#     boundary (a step is well under a second on A100/H200/L40S) and exits 3; the
#     requeued task finds last.pt and resumes mid-epoch with an identical data order.
#     The trainer also writes last.pt every 20 min and at every epoch end, which bounds
#     the loss if SIGKILL ever wins the race.
#   * DefaultTime=04:00:00, so --time is always set. --signal=B:USR1@300 warns the batch
#     shell 5 min before the limit: the trainer checkpoints and this script requeues the
#     task with `scontrol requeue` (a plain SIGTERM is NOT requeued here, so scancel works).
#   * --constraint a100|h200|l40s: the a40 nodes carry no feature tag and the
#     rtx_pro_6000 (sm_120) nodes cannot run torch 2.4.1+cu121. The trainer also refuses
#     an unsupported GPU with exit 4.
#
# SAFETY
#   * a task whose output already holds results.json is skipped;
#   * an atomic mkdir lock (pattern of sh/_eeg250.sh) stops two runners training into the
#     same output dir; a requeued task (same SLURM_JOB_ID) reclaims its own lock;
#   * nothing is ever deleted: last.pt / best_*.pt / val_ep*.npz all stay.
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
TABLE=${1:-$GRADER_DIR/stage1.tsv}               # resolved before the cd below
case "$TABLE" in /*) ;; *) if [ -f "$TABLE" ]; then TABLE=$PWD/$TABLE
                           elif [ -f "$GRADER_DIR/$TABLE" ]; then TABLE=$GRADER_DIR/$TABLE
                           else TABLE=$PWD/$TABLE; fi ;; esac   # never left relative: the cd re-anchors it
cd "$EEG_ROOT" || exit 1                         # the trainer's relative paths are EEG_ROOT's
ENV=/work/mech-ai-scratch/alloy/.conda/envs/eeg
export PATH="$ENV/bin:$PATH"
export TORCH_HOME=/work/mech-ai-scratch/alloy/.cache/torch
export HF_HOME=/work/mech-ai-scratch/alloy/.cache/hf
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
L=${TTG_LOG_DIR:-$EEG_ROOT/logs/ttg}      # override only for local tests
log(){ echo "[$(date +%F_%T)] $*"; }

i=${SLURM_ARRAY_TASK_ID:-0}
[ -f "$TABLE" ] || { log "no config table $TABLE"; exit 1; }
line=$(grep -vE '^[[:space:]]*(#|$)' "$TABLE" | sed -n "$((i + 1))p")
[ -n "$line" ] || { log "no line $i in $TABLE"; exit 1; }
read -r name arch fix heads seed split fold out extra <<< "$line"
extra=${extra:-}
[ "$extra" = "-" ] && extra=""

case "$arch" in x3d) CACHE=cache_frames/f16s224 ;; slowfast) CACHE=cache_frames/f32s224 ;;
  *) log "bad arch '$arch' in line $i"; exit 1 ;; esac
CACHE=${TTG_CACHE:-$CACHE}                   # override only for local tests
case "$fix" in fixed) FIX="--fix_x3d" ;; bugged) FIX="" ;;
  *) log "bad fix '$fix' (fixed|bugged) in line $i"; exit 1 ;; esac
case "$heads" in dual|g3|g5) ;; *) log "bad heads '$heads' in line $i"; exit 1 ;; esac
case "$split" in session|subject) ;; *) log "bad split '$split' in line $i"; exit 1 ;; esac
case "$out" in /*) out_real=$(realpath -m -- "$out") ;; *) out_real="" ;; esac   # symlinks resolved
case "$out_real" in "$EEG_ROOT"/output/ttg_*) ;; *) log "output must be absolute under $EEG_ROOT/output/ttg_*: $out"; exit 1 ;; esac

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} - 4 ))
CMD=(python "$GRADER_DIR/train_grader.py" --arch "$arch" $FIX --heads "$heads" --seed "$seed"
     --split "$split" --split_seed 49 --fold "$fold" --epochs 12 --lr 1e-4 --batch_size 8
     --workers "$WORKERS" --cache_dir "$CACHE" --output "$out" $extra)

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
[ -f "$CACHE/index.json" ] || { log "PREFLIGHT FAILED: $CACHE/index.json missing (build it with $GRADER_DIR/sbatch_cache.sh)"; exit 1; }
python - "$CACHE/index.json" <<'PY' || { log "PREFLIGHT FAILED: $CACHE lists unreadable clips; the trainer would refuse them"; exit 1; }
import json, sys
u = json.load(open(sys.argv[1])).get("unreadable", [])
if u:
    print(f"{len(u)} unreadable clips in {sys.argv[1]}, e.g. {u[0]}")
sys.exit(1 if u else 0)
PY
python -c "import torch, pytorchvideo, cv2, sklearn" || { log "PREFLIGHT FAILED: env imports"; exit 1; }

log "START $name (job ${SLURM_JOB_ID:-local} task $i on $(hostname)); log $L/$name.log"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
log "CMD ${CMD[*]}"

# ---- run, forwarding preemption / time-limit signals to the trainer -----------------------
child=""
usr1=0
fwd(){ [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; }
trap 'log "SIGTERM (preemption or scancel): forwarding to trainer"; fwd' TERM
trap 'log "SIGUSR1 (time limit in 300 s): checkpoint, then requeue"; usr1=1; fwd' USR1
"${CMD[@]}" >> "$L/$name.log" 2>&1 &
child=$!
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break      # still alive (or unreaped): wait again
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
  log "FAILED $name (exit $rc); tail of $L/$name.log:"; tail -5 "$L/$name.log"; exit 1
fi
