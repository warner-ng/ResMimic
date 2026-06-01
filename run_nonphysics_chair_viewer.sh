#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"

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
python "$RESMIMIC_ROOT/scripts/play_nonphysics_viser.py" \
  --human "$RESMIMIC_ROOT/legged_gym/assets/motions/Date03_Sub01_bike_on_may_29_21_17_human_upright_bikez.pkl" \
  --object "$RESMIMIC_ROOT/legged_gym/assets/motions/Date03_Sub01_bike_on_may_29_21_17_object_upright_bikez.npz" \
  --robot-urdf "$RESMIMIC_ROOT/assets/g1/g1_custom_collision_29dof.urdf" \
  --object-urdf "$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/hy3d_bike_cari4d.urdf" \
  --host 0.0.0.0 \
  --port 8080
