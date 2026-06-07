#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"
require_env() {
  local var="$1"
  : "${!var:?$var is required but not set for nonphysics viewer}"
}
require_env PRE_HUMAN
require_env POST_HUMAN
require_env OBJECT_MOTION
require_env ALIGNED_HUMAN
require_env ALIGN_HUMAN_YAW_TO_OBJECT
require_env OBJECT_SCALE
require_env OBJECT_MESH
require_env OBJECT_URDF
require_env HUMAN_ROOT_ROT_ROLL_DEG
require_env HUMAN_ROOT_ROT_PITCH_DEG
require_env HUMAN_ROOT_ROT_YAW_DEG
require_env OBJECT_ROOT_ROT_ROLL_DEG
require_env OBJECT_ROOT_ROT_PITCH_DEG
require_env OBJECT_ROOT_ROT_YAW_DEG
require_env OBJECT_ROOT_LOCAL_ROT_ROLL_DEG
require_env OBJECT_ROOT_LOCAL_ROT_PITCH_DEG
require_env OBJECT_ROOT_LOCAL_ROT_YAW_DEG
require_env OBJECT_ROOT_POS_OFFSET_X
require_env OBJECT_ROOT_POS_OFFSET_Y
require_env OBJECT_ROOT_POS_OFFSET_Z
require_env ENABLE_RUNTIME_PAIR_LEVELING
require_env RUNTIME_PAIR_LEVEL_TARGET_Z
require_env HUMAN_ROOT_Z_BIAS
require_env OBJECT_ROOT_Z_BIAS
free_tcp_port() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 { if ($NF ~ /pid=/) { match($NF, /pid=([0-9]+)/, a); if (a[1] != "") print a[1] } }')"
  fi

  if [[ -n "$pids" ]]; then
    echo "[WARN] 端口 $port 已占用，先回收 PID(s): $(echo "$pids" | tr '\n' ' ')"
    echo "$pids" | xargs -r kill -15 || true
    sleep 1
    if command -v lsof >/dev/null 2>&1; then
      pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v ss >/dev/null 2>&1; then
      pids="$(ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 { if ($NF ~ /pid=/) { match($NF, /pid=([0-9]+)/, a); if (a[1] != "") print a[1] } }')"
    fi
    if [[ -n "$pids" ]]; then
      echo "[WARN] 强制清理端口 $port"
      echo "$pids" | xargs -r kill -9 || true
      sleep 1
    fi
  fi

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 { if ($NF ~ /pid=/) { match($NF, /pid=([0-9]+)/, a); if (a[1] != "") print a[1] } }')"
  fi
  if [[ -n "$pids" ]]; then
    die "端口 $port 仍被占用，请手动释放后重试。"
  fi
}

source "$RESMIMIC_ROOT/source_dev_setup.sh"
cd "$RESMIMIC_ROOT"

echo "[1/3] Compare pre-GMR vs post-GMR root motion"
python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$POST_HUMAN" \
  --object "$OBJECT_MOTION" \
  --quiet --topk 0

echo
echo "[2/3] Build rigidly aligned post-GMR human motion"
ALIGN_ARGS=()
if [[ "$ALIGN_HUMAN_YAW_TO_OBJECT" == "1" ]]; then
  ALIGN_ARGS+=(--align-yaw)
fi
python "$RESMIMIC_ROOT/scripts/align_human_root_to_object.py" \
  --human "$POST_HUMAN" \
  --object "$OBJECT_MOTION" \
  --output "$ALIGNED_HUMAN" \
  --quiet \
  "${ALIGN_ARGS[@]}"

echo
echo "[2.5/3] Re-check aligned human vs object root motion"
python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION" \
  --quiet --topk 0

echo
echo "[3/3] Launch viser preview with aligned human motion"
free_tcp_port 8080

if [[ "${OBJECT_SCALE}" != "1.0" ]]; then
  echo "[INFO] OBJECT_SCALE=${OBJECT_SCALE}：viser 使用运行时scale参数，不生成缩放文件。"
fi

LEVEL_ARGS=()
if [[ "$ENABLE_RUNTIME_PAIR_LEVELING" == "1" ]]; then
  LEVEL_ARGS+=(--enable-runtime-pair-leveling)
fi

python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
  --human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION" \
  --object-mesh "$OBJECT_MESH" \
  --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
  --object-urdf "$OBJECT_URDF" \
  --human-root-rot-roll-deg "$HUMAN_ROOT_ROT_ROLL_DEG" \
  --human-root-rot-pitch-deg "$HUMAN_ROOT_ROT_PITCH_DEG" \
  --human-root-rot-yaw-deg "$HUMAN_ROOT_ROT_YAW_DEG" \
  --object-root-rot-roll-deg "$OBJECT_ROOT_ROT_ROLL_DEG" \
  --object-root-rot-pitch-deg "$OBJECT_ROOT_ROT_PITCH_DEG" \
  --object-root-rot-yaw-deg "$OBJECT_ROOT_ROT_YAW_DEG" \
  --object-root-local-rot-roll-deg "$OBJECT_ROOT_LOCAL_ROT_ROLL_DEG" \
  --object-root-local-rot-pitch-deg "$OBJECT_ROOT_LOCAL_ROT_PITCH_DEG" \
  --object-root-local-rot-yaw-deg "$OBJECT_ROOT_LOCAL_ROT_YAW_DEG" \
  --object-root-pos-offset-x "$OBJECT_ROOT_POS_OFFSET_X" \
  --object-root-pos-offset-y "$OBJECT_ROOT_POS_OFFSET_Y" \
  --object-root-pos-offset-z "$OBJECT_ROOT_POS_OFFSET_Z" \
  --object-mesh-scale "$OBJECT_SCALE" \
  "${LEVEL_ARGS[@]}" \
  --runtime-pair-level-target-z "$RUNTIME_PAIR_LEVEL_TARGET_Z" \
  --human-root-z-bias "$HUMAN_ROOT_Z_BIAS" \
  --object-root-z-bias "$OBJECT_ROOT_Z_BIAS" \
  --host 0.0.0.0 \
  --port 8080
