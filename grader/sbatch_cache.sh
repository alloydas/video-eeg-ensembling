#!/bin/bash
#SBATCH --job-name=ttg-f16cache
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=/work/mech-ai-scratch/alloy/EEG/logs/ttg/f16cache_%j.out
#
# ==========================================================================================
#  CPU job: rebuild cache_frames/f16s224 from data_full with grader/build_f16_cache.py.
# ==========================================================================================
#
# PURPOSE
#   Stage 1 trains X3D-M on the historical 16x224 linspace input. Only f32s224 survives on
#   Nova, and data/ holds zero-byte placeholders, so the 16-frame cache is re-decoded from
#   data_full/ (keys stay data/...). ~24,497 clips, ~55 GiB. Measured ~13-16 clips/s with
#   4-6 workers on a loaded login-class node; 30 workers should finish well inside 6 h.
#   The builder is RESUMABLE: if this job hits the time limit, resubmit it and it continues
#   from progress.npz. It aborts (no index written) if any source is zero-byte/missing or
#   more than 0.5% of clips fail, and verifies 50 random clips bit-for-bit against
#   train_classifier.load_clip before publishing index.json. This job then exits 1 if the
#   published index lists ANY unreadable clip (train_grader.py would refuse to start), so
#   `--dependency=afterok` only releases Stage 1 on a complete cache.
#
# WHERE THINGS ARE
#   The builder is $GRADER_DIR/build_f16_cache.py (video-eeg-ensembling/grader; under sbatch
#   "$0" is a spooled copy, so $GRADER_DIR, else the checkout sbatch was run from, else the
#   fixed default is used); it runs with
#   cwd = EEG_ROOT (env var, default /work/mech-ai-scratch/alloy/EEG) and writes the cache to
#   $EEG_ROOT/cache_frames/f16s224.
#
# USAGE
#   mkdir -p /work/mech-ai-scratch/alloy/EEG/logs/ttg      # SLURM will not create it
#   (or: bash grader/submit_ttg.sh stage1, which makes the directory, submits this job and
#    chains the Stage-1 array on it with afterok)
#   cd /work/mech-ai-scratch/alloy/video-eeg-ensembling && sbatch grader/sbatch_cache.sh
#   then:  sbatch --dependency=afterok:<cache job id> --array=0-10 grader/sbatch_grader.sh grader/stage1.tsv
#   DRY_RUN=1 bash grader/sbatch_cache.sh   # prints the plan (runs the builder with --dry_run)
#
# nova is PreemptMode=GANG,SUSPEND (no progress lost) and DefaultTime 04:00:00, so --time
# is set explicitly. Disk: /work/mech-ai-scratch had 3.9 TB free when this was written.
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
cd "$EEG_ROOT" || exit 1                         # the builder's relative paths are EEG_ROOT's
ENV=/work/mech-ai-scratch/alloy/.conda/envs/eeg
export PATH="$ENV/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
OUT=${TTG_CACHE_OUT:-cache_frames/f16s224}     # override only for local tests of the index check
W=$(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))
[ "$W" -ge 1 ] || W=1
if [ "${DRY_RUN:-0}" = "1" ]; then
  exec python "$GRADER_DIR/build_f16_cache.py" --out "$OUT" --workers "$W" --dry_run
fi
# The builder publishes index.json with up to 0.5% 'unreadable' clips and exits 0, but
# train_grader.py refuses to start if ANY train/val clip is unreadable. So this job exits 1
# unless the published index lists none: an afterok dependency then never releases a Stage-1
# array whose 11 tasks would all exit 1 at startup.
check_index(){
  python - "$OUT/index.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
u = m.get("unreadable", [])
print(f"index: {m['n']} clips, {m['frames']}x{m['size']}, unreadable {len(u)}")
for p in u[:20]:
    print("   UNREADABLE", p, "--", m.get("unreadable_reasons", {}).get(p, ""))
if u:
    print("NOT READY for train_grader.py (it refuses any unreadable train/val clip): fix the "
          "sources and rebuild with --overwrite, or decide on a drop policy first")
sys.exit(1 if u else 0)
PY
}
if [ -f "$OUT/index.json" ]; then
  echo "[$(date +%F_%T)] SKIP build: $OUT/index.json already exists"
  check_index; exit $?
fi
echo "[$(date +%F_%T)] START f16s224 build on $(hostname) with $W workers (job ${SLURM_JOB_ID:-local})"
python "$GRADER_DIR/build_f16_cache.py" --out "$OUT" --workers "$W" --verify 50
rc=$?
echo "[$(date +%F_%T)] END rc=$rc"
[ "$rc" -eq 0 ] || exit $rc
check_index; rc=$?
[ "$rc" -eq 0 ] && echo "[$(date +%F_%T)] cache READY: 0 unreadable clips"
exit $rc
