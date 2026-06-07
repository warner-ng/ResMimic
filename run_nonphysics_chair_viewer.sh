#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"
ALIGN_HUMAN_YAW_TO_OBJECT="${ALIGN_HUMAN_YAW_TO_OBJECT:-0}"
HUMAN_ROOT_ROT_ROLL_DEG="${HUMAN_ROOT_ROT_ROLL_DEG:-0.0}"
HUMAN_ROOT_ROT_PITCH_DEG="${HUMAN_ROOT_ROT_PITCH_DEG:-0.0}"
HUMAN_ROOT_ROT_YAW_DEG="${HUMAN_ROOT_ROT_YAW_DEG:-0.0}"
OBJECT_ROOT_ROT_ROLL_DEG="${OBJECT_ROOT_ROT_ROLL_DEG:-0.0}"
OBJECT_ROOT_ROT_PITCH_DEG="${OBJECT_ROOT_ROT_PITCH_DEG:-0.0}"
OBJECT_ROOT_ROT_YAW_DEG="${OBJECT_ROOT_ROT_YAW_DEG:-0.0}"
OBJECT_ROOT_LOCAL_ROT_ROLL_DEG="${OBJECT_ROOT_LOCAL_ROT_ROLL_DEG:-0.0}"
OBJECT_ROOT_LOCAL_ROT_PITCH_DEG="${OBJECT_ROOT_LOCAL_ROT_PITCH_DEG:-0.0}"
OBJECT_ROOT_LOCAL_ROT_YAW_DEG="${OBJECT_ROOT_LOCAL_ROT_YAW_DEG:-0.0}"
OBJECT_ROOT_POS_OFFSET_X="${OBJECT_ROOT_POS_OFFSET_X:-0.0}"
OBJECT_ROOT_POS_OFFSET_Y="${OBJECT_ROOT_POS_OFFSET_Y:-0.0}"
OBJECT_ROOT_POS_OFFSET_Z="${OBJECT_ROOT_POS_OFFSET_Z:-0.0}"
ENABLE_RUNTIME_PAIR_LEVELING="${ENABLE_RUNTIME_PAIR_LEVELING:-0}"
RUNTIME_PAIR_LEVEL_TARGET_Z="${RUNTIME_PAIR_LEVEL_TARGET_Z:-0.0}"
HUMAN_ROOT_Z_BIAS="${HUMAN_ROOT_Z_BIAS:-0.05}"
OBJECT_ROOT_Z_BIAS="${OBJECT_ROOT_Z_BIAS:-0.03}"
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
LEVEL_ARGS=()
if [[ "$ENABLE_RUNTIME_PAIR_LEVELING" == "1" ]]; then
  LEVEL_ARGS+=(--enable-runtime-pair-leveling)
fi

python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
  --human "$ALIGNED_HUMAN" \
  --object "$OBJECT_MOTION" \
  --object-mesh "$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/Date03_Sub01_bike_wild001_000_align_centered_x20.obj" \
  --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
  --object-urdf "$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/hy3d_bike_cari4d.urdf" \
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
  "${LEVEL_ARGS[@]}" \
  --runtime-pair-level-target-z "$RUNTIME_PAIR_LEVEL_TARGET_Z" \
  --human-root-z-bias "$HUMAN_ROOT_Z_BIAS" \
  --object-root-z-bias "$OBJECT_ROOT_Z_BIAS" \
  --host 0.0.0.0 \
  --port 8080
