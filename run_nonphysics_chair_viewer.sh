#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"
OBJECT_VIEWER_SCALE="${OBJECT_VIEWER_SCALE:-1.0}"
OBJECT_MESH_MIRROR_AXIS="${OBJECT_MESH_MIRROR_AXIS:-none}"
OBJECT_SCALE_DEBUG_CUBE="${OBJECT_SCALE_DEBUG_CUBE:-0}"
OBJECT_SCALE_MOTION_FALLBACK="${OBJECT_SCALE_MOTION_FALLBACK:-0}"
ALIGN_HUMAN_YAW_TO_OBJECT="${ALIGN_HUMAN_YAW_TO_OBJECT:-0}"
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
PRE_HUMAN="${PRE_HUMAN:-$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_smplx_input.npz}"
POST_HUMAN="${POST_HUMAN:-$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez.pkl}"
OBJECT_MOTION="${OBJECT_MOTION:-$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_object_upright_bikez.npz}"
ALIGNED_HUMAN="${ALIGNED_HUMAN:-$RESMIMIC_ROOT/assets/motions/Date03_Sub01_bike_May_31_19_34_human_upright_bikez_aligned.pkl}"

echo "[1/3] Compare pre-GMR vs post-GMR root motion"
python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$POST_HUMAN" \
  --object "$OBJECT_MOTION"

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
  "${ALIGN_ARGS[@]}"

echo
echo "[2.5/3] Re-check aligned human vs object root motion"
python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION"

echo
echo "[3/3] Launch viser preview with aligned human motion"
EXTRA_ARGS=()
if [[ "$OBJECT_SCALE_DEBUG_CUBE" == "1" ]]; then
  EXTRA_ARGS+=(--debug-object-scale-cube)
fi
if [[ "$OBJECT_SCALE_MOTION_FALLBACK" == "1" ]]; then
  EXTRA_ARGS+=(--object-motion-scale-fallback)
fi

python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
  --human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION" \
  --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
  --object-urdf "$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/hy3d_bike_cari4d.urdf" \
  --object-scale "$OBJECT_VIEWER_SCALE" \
  --object-mesh-mirror-axis "$OBJECT_MESH_MIRROR_AXIS" \
  --host 0.0.0.0 \
  --port 8080 \
  "${EXTRA_ARGS[@]}"
