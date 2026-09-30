#!/bin/bash
# The EEG-gated grader on the continuous recordings (scan_false_alarms.py; pre-registration scan_prereg.md).
# Outputs go only to $EEG_ROOT/output/ttg_scan/. Every stage is resumable; jobs are requeue-safe.
#
#   bash submit_scan.sh cpu <command> [args...]      one CPU scavenger job running a sub-command (e.g. stepa-prep)
#   bash submit_scan.sh decode [N] [args...]         N parallel CPU scavenger decode jobs (default 6), each takes
#                                                    every N-th unit (--index i --stride N); --pilot for S3
#   bash submit_scan.sh gpu <command> <HH:MM:SS> [args...]   one GPU scavenger job (stepa-gpu | infer), refused
#                                                    when the GPU ledger plus its --time would exceed 6.0 h
#   bash submit_scan.sh ledger                       GPU hours used, from budget/gpu.jsonl reconciled with sacct
#
# Stages actually run (2026-09-29/30, see ttg_scan/*/run.json):
#   S0 tables + gates  on the interactive allocation (numpy / sklearn; minutes)
#   S1 boxes           on the interactive allocation (decodes 3 x 3 days of keyframe pairs)
#   S2 stepa-prep (cpu), stepa-gpu (gpu), stepa-eval (interactive)
#   S3/S4 decode (cpu array) + clock (cpu) + infer (gpu), S5 analyse, S6 gallery
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/work/mech-ai-scratch/alloy/.conda/envs/eeg/bin/python}"     # torch, av, cv2, sklearn
export EEG_ROOT="${EEG_ROOT:-/work/mech-ai-scratch/alloy/EEG}"
OUT="$EEG_ROOT/output/ttg_scan"
mkdir -p "$OUT/logs" "$OUT/budget"
UNSET=$(env | awk -F= '/^SLURM_/{printf "-u %s ", $1}')
SB="env $UNSET sbatch -A mech-ai-scavenger -q scavenger -p scavenger --requeue --export=ALL"
export PYTHONDONTWRITEBYTECODE=1
S="$HERE/scan_false_alarms.py"
MODE="${1:?mode}"; shift

gpu_hours_used() {   # completed GPU jobs: sacct elapsed; running / pending ones: their time limit
  local ids
  ids=$(grep -o '"job": "[0-9]*"' "$OUT/budget/gpu_jobs.jsonl" 2>/dev/null | grep -o '[0-9]*' | sort -u | tr '\n' ',' || true)
  if [[ -z "$ids" ]]; then echo 0; return; fi
  env $UNSET sacct -X -D -n -P -j "${ids%,}" -o JobID,State,ElapsedRaw,TimelimitRaw | awk -F'|' '
    { if ($2 ~ /RUNNING|PENDING|REQUEUED/) s += $4 * 60; else s += $3 }
    END { printf "%.3f\n", s / 3600 }'
}

case "$MODE" in
  cpu)
    CMD="${1:?command}"; shift
    $SB --parsable -c "${CPUS:-8}" --mem="${MEM:-32G}" -t "${TIME:-04:00:00}" -J "scan_$CMD" \
        -o "$OUT/logs/${CMD}_%j.out" \
        --wrap "OMP_NUM_THREADS=${CPUS:-8} $PY $S $CMD $*"
    ;;
  motion)
    N="${1:-8}"; shift || true
    for ((i = 0; i < N; i++)); do
      $SB --parsable -c "${CPUS:-8}" --mem="${MEM:-16G}" -t "${TIME:-08:00:00}" -J "scan_mot$i" \
          -o "$OUT/logs/motion_${i}_%j.out" \
          --wrap "OMP_NUM_THREADS=2 $PY $S motion --index $i --stride $N --threads ${CPUS:-8} $*"
    done
    ;;
  decode)
    N="${1:-6}"; shift || true
    for ((i = 0; i < N; i++)); do
      $SB --parsable -c "${CPUS:-8}" --mem="${MEM:-48G}" -t "${TIME:-12:00:00}" -J "scan_dec$i" \
          -o "$OUT/logs/decode_${i}_%j.out" \
          --wrap "OMP_NUM_THREADS=2 $PY $S decode --index $i --stride $N --threads ${CPUS:-8} $*"
    done
    ;;
  gpu)
    CMD="${1:?command}"; TL="${2:?time limit HH:MM:SS}"; shift 2
    IFS=: read -r hh mm ss <<< "$TL"
    NEW=$(awk -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.3f", h + m / 60 + s / 3600 }')
    USED=$(gpu_hours_used)
    if awk -v u="$USED" -v n="$NEW" 'BEGIN { exit !(u + n > 6.0) }'; then
      echo "REFUSED: GPU ledger $USED h + this job's $NEW h > 6.0 h" >&2; exit 2
    fi
    DEADLINE_S=$(awk -v n="$NEW" 'BEGIN { printf "%d", n * 3600 - 420 }')   # stop starting shards 7 min before
    JID=$($SB --parsable -c 16 --mem=96G -t "$TL" --gres=gpu:1 --constraint="a100|h200|l40s" -J "scan_$CMD" \
        -o "$OUT/logs/gpu_${CMD}_%j.out" --signal=B:USR1@300 \
        --wrap "T0=\$(date +%s.%N); echo prewarm start \$(date); \
                xargs -a $OUT/budget/prewarm_files.txt -d '\n' -P 32 -n 16 cat > /dev/null 2>&1 || true; \
                echo prewarm done \$(date); \
                exec $PY $S $CMD --t_launch \$T0 --deadline_s $DEADLINE_S $*")
    echo "{\"job\": \"$JID\", \"cmd\": \"$CMD\", \"time_limit_h\": $NEW, \"submitted\": \"$(date '+%F %T')\", \"ledger_before_h\": $USED}" >> "$OUT/budget/gpu_jobs.jsonl"
    echo "$JID"
    ;;
  scan)
    # The fused GPU job (scan_false_alarms.py scan-gpu): step A if not decided, the S3 pilot, then S4 with decoding
    # on the job's own CPUs into node-local /dev/shm. The eeg env and the heavy inputs are streamed over ssh from a
    # node-local stage on the interactive node (STAGE_HOST:STAGE_DIR: env.tar + data.tar, built from its page
    # cache), because cold reads from /work/mech-ai-scratch ran at 1-5 MB/s on 2026-09-29 (a first GPU job spent
    # 34 min importing and was cancelled). Refused when the GPU ledger plus --time exceeds 6.0 h.
    TL="${1:?time limit HH:MM:SS}"; shift
    STAGE_HOST="${STAGE_HOST:?stage host}"; STAGE_DIR="${STAGE_DIR:?stage dir on the stage host}"
    IFS=: read -r hh mm ss <<< "$TL"
    NEW=$(awk -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.3f", h + m / 60 + s / 3600 }')
    USED=$(gpu_hours_used)
    if awk -v u="$USED" -v n="$NEW" 'BEGIN { exit !(u + n > 6.0) }'; then
      echo "REFUSED: GPU ledger $USED h + this job's $NEW h > 6.0 h" >&2; exit 2
    fi
    DEADLINE_S=$(awk -v n="$NEW" 'BEGIN { printf "%d", n * 3600 - 480 }')
    ENVP=/work/mech-ai-scratch/alloy/.conda/envs/eeg
    JID=$($SB --parsable -c "${CPUS:-32}" --mem="${MEM:-160G}" -t "$TL" --gres=gpu:1 --constraint="a100|h200|l40s" \
        -J scan_gpu -o "$OUT/logs/gpu_scan_%j.out" --signal=B:USR1@300 \
        --wrap "T0=\$(date +%s.%N); R=\${TMPDIR:-/tmp}/scanroot_\$SLURM_JOB_ID; mkdir -p \$R$ENVP; \
                SSH='ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20'; \
                PYX=\$R$ENVP/bin/python; export SCAN_STAGE=\$R; \
                if \$SSH $STAGE_HOST cat $STAGE_DIR/env.tar | tar -x -C \$R$ENVP && \$SSH $STAGE_HOST cat $STAGE_DIR/data.tar | tar -x -C \$R; then \
                  echo staged \$(du -sh \$R | cut -f1) in \$(awk -v a=\$T0 -v b=\$(date +%s.%N) 'BEGIN{print b-a}') s; \
                else echo STAGING FAILED: using the env and inputs on /work/mech-ai-scratch directly; PYX=$ENVP/bin/python; unset SCAN_STAGE; fi; \
                export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4; \
                \$PYX $S scan-gpu --t_launch \$T0 --deadline_s $DEADLINE_S --workers ${WORKERS:-7} --threads 4 $* & \
                P=\$!; trap 'kill -USR1 \$P' USR1; trap 'kill -TERM \$P' TERM; \
                wait \$P; rc=\$?; while kill -0 \$P 2>/dev/null; do wait \$P; rc=\$?; done; rm -rf \$R; exit \$rc")
    echo "{\"job\": \"$JID\", \"cmd\": \"scan-gpu\", \"time_limit_h\": $NEW, \"submitted\": \"$(date '+%F %T')\", \"ledger_before_h\": $USED}" >> "$OUT/budget/gpu_jobs.jsonl"
    echo "$JID"
    ;;
  ledger)
    echo "GPU hours used (sacct elapsed of finished jobs + time limit of running / pending ones): $(gpu_hours_used)"
    ;;
  *)
    echo "unknown mode $MODE" >&2; exit 1 ;;
esac
