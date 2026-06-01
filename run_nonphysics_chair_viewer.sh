#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"
OBJECT_VIEWER_SCALE="${OBJECT_VIEWER_SCALE:-1.0}"
OBJECT_MESH_MIRROR_AXIS="${OBJECT_MESH_MIRROR_AXIS:-none}"
OBJECT_SCALE_DEBUG_CUBE="${OBJECT_SCALE_DEBUG_CUBE:-0}"
OBJECT_SCALE_MOTION_FALLBACK="${OBJECT_SCALE_MOTION_FALLBACK:-0}"
OBJECT_RPY_ROLL_DEG="${OBJECT_RPY_ROLL_DEG:-0.0}"
OBJECT_RPY_PITCH_DEG="${OBJECT_RPY_PITCH_DEG:-0.0}"
OBJECT_RPY_YAW_DEG="${OBJECT_RPY_YAW_DEG:-0.0}"

source "$RESMIMIC_ROOT/source_dev_setup.sh"
cd "$RESMIMIC_ROOT"

# Chair version (kept for reference)
# python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
#   --human "$RESMIMIC_ROOT/legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_human_upright_suitcasez.pkl" \
#   --object "$RESMIMIC_ROOT/legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_object_upright_suitcasez.npz" \
#   --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
#   --object-urdf "$RESMIMIC_ROOT/legged_gym/assets/chairblack_cari4d/chairblack_cari4d.urdf" \
#   --host 0.0.0.0 \
#   --port 8080

# Bike version
EXTRA_ARGS=()
if [[ "$OBJECT_SCALE_DEBUG_CUBE" == "1" ]]; then
  EXTRA_ARGS+=(--debug-object-scale-cube)
fi
if [[ "$OBJECT_SCALE_MOTION_FALLBACK" == "1" ]]; then
  EXTRA_ARGS+=(--object-motion-scale-fallback)
fi

python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
  --human "$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez.pkl" \
  --object "$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez.npz" \
  --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
  --object-urdf "$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/hy3d_bike_cari4d.urdf" \
  --object-scale "$OBJECT_VIEWER_SCALE" \
  --object-mesh-mirror-axis "$OBJECT_MESH_MIRROR_AXIS" \
  --object-rpy-roll-deg "$OBJECT_RPY_ROLL_DEG" \
  --object-rpy-pitch-deg "$OBJECT_RPY_PITCH_DEG" \
  --object-rpy-yaw-deg "$OBJECT_RPY_YAW_DEG" \
  --host 0.0.0.0 \
  --port 8080 \
  "${EXTRA_ARGS[@]}"
