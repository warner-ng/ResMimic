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
require_env OBJECT_SCALE
require_env OBJECT_MESH
require_env OBJECT_URDF
require_env HUMAN_ROOT_ROT_ROLL_DEG
require_env HUMAN_ROOT_ROT_PITCH_DEG
require_env HUMAN_ROOT_ROT_YAW_DEG
require_env OBJECT_ROOT_ROT_ROLL_DEG
require_env OBJECT_ROOT_ROT_PITCH_DEG
require_env OBJECT_ROOT_ROT_YAW_DEG
require_env OBJECT_ROOT_POS_OFFSET_X
require_env OBJECT_ROOT_POS_OFFSET_Y
require_env OBJECT_ROOT_POS_OFFSET_Z
require_env ENABLE_RUNTIME_PAIR_LEVELING
require_env RUNTIME_PAIR_LEVEL_TARGET_Z
require_env HUMAN_ROOT_Z_BIAS
require_env OBJECT_ROOT_Z_BIAS
require_env HUMAN_OBJECT_ROOT_TRANS_X
require_env HUMAN_OBJECT_ROOT_TRANS_Y
require_env HUMAN_OBJECT_ROOT_TRANS_Z
VIEWER_PORT="${VIEWER_PORT:-8080}"

HUMAN_OBJECT_ROOT_ROT_ROLL_DEG="${HUMAN_OBJECT_ROOT_ROT_ROLL_DEG:-0.0}"
HUMAN_OBJECT_ROOT_ROT_PITCH_DEG="${HUMAN_OBJECT_ROOT_ROT_PITCH_DEG:-0.0}"
HUMAN_OBJECT_ROOT_ROT_YAW_DEG="${HUMAN_OBJECT_ROOT_ROT_YAW_DEG:-0.0}"

sum_deg() {
  local a="$1"
  local b="$2"
  awk "BEGIN { printf \"%.6f\", $a + $b }"
}

HUMAN_ROOT_ROT_ROLL_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_ROLL_DEG" "$HUMAN_ROOT_ROT_ROLL_DEG")"
HUMAN_ROOT_ROT_PITCH_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_PITCH_DEG" "$HUMAN_ROOT_ROT_PITCH_DEG")"
HUMAN_ROOT_ROT_YAW_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_YAW_DEG" "$HUMAN_ROOT_ROT_YAW_DEG")"
OBJECT_ROOT_ROT_ROLL_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_ROLL_DEG" "$OBJECT_ROOT_ROT_ROLL_DEG")"
OBJECT_ROOT_ROT_PITCH_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_PITCH_DEG" "$OBJECT_ROOT_ROT_PITCH_DEG")"
OBJECT_ROOT_ROT_YAW_DEG_COMBINED="$(sum_deg "$HUMAN_OBJECT_ROOT_ROT_YAW_DEG" "$OBJECT_ROOT_ROT_YAW_DEG")"

tcp_port_in_use() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 { if ($NF ~ /pid=/) { match($NF, /pid=([0-9]+)/, a); if (a[1] != "") print a[1] } }')"
  fi

  if [[ -n "$pids" ]]; then
    return 0
  fi
  return 1
}

find_available_tcp_port() {
  local port="${1:-8080}"
  local max_port="${2:-8099}"

  while (( port <= max_port )); do
    if ! tcp_port_in_use "$port"; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done

  die "未找到可用 viewer 端口（检查范围 ${1:-8080}-${max_port}）。"
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
python "$RESMIMIC_ROOT/scripts/align_human_root_to_object.py" \
  --human "$POST_HUMAN" \
  --object "$OBJECT_MOTION" \
  --output "$ALIGNED_HUMAN" \
  --quiet

echo
echo "[2.5/3] Re-check aligned human vs object root motion"
python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION" \
  --quiet --topk 0

echo
echo "[3/3] Launch viser preview with aligned human motion"
VIEWER_PORT="$(find_available_tcp_port "$VIEWER_PORT" 8099)"
echo "[INFO] nonphysics viewer port=$VIEWER_PORT"

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
  --human-root-rot-roll-deg "$HUMAN_ROOT_ROT_ROLL_DEG_COMBINED" \
  --human-root-rot-pitch-deg "$HUMAN_ROOT_ROT_PITCH_DEG_COMBINED" \
  --human-root-rot-yaw-deg "$HUMAN_ROOT_ROT_YAW_DEG_COMBINED" \
  --object-root-rot-roll-deg "$OBJECT_ROOT_ROT_ROLL_DEG_COMBINED" \
  --object-root-rot-pitch-deg "$OBJECT_ROOT_ROT_PITCH_DEG_COMBINED" \
  --object-root-rot-yaw-deg "$OBJECT_ROOT_ROT_YAW_DEG_COMBINED" \
  --object-root-pos-offset-x "$OBJECT_ROOT_POS_OFFSET_X" \
  --object-root-pos-offset-y "$OBJECT_ROOT_POS_OFFSET_Y" \
  --object-root-pos-offset-z "$OBJECT_ROOT_POS_OFFSET_Z" \
  --pair-root-pos-offset-x "$HUMAN_OBJECT_ROOT_TRANS_X" \
  --pair-root-pos-offset-y "$HUMAN_OBJECT_ROOT_TRANS_Y" \
  --pair-root-pos-offset-z "$HUMAN_OBJECT_ROOT_TRANS_Z" \
  --object-mesh-scale "$OBJECT_SCALE" \
  "${LEVEL_ARGS[@]}" \
  --runtime-pair-level-target-z "$RUNTIME_PAIR_LEVEL_TARGET_Z" \
  --human-root-z-bias "$HUMAN_ROOT_Z_BIAS" \
  --object-root-z-bias "$OBJECT_ROOT_Z_BIAS" \
  --show-ground \
  --ground-size 12 \
  --ground-cell-size 0.5 \
  --host 0.0.0.0 \
  --port "$VIEWER_PORT"
