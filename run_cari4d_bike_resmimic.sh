#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# CARI4D Bike -> ResMimic (g1_hoi_bike_cari4d)
# 按“给定 output .pth + 任务名”的一键流程脚本
#
# ===== 你主要改这几行（按优先级）=====
# 1) CARI4D_PTH
# 2) TAG
# 3) PAIR_SUFFIX（通常 bikez）

########################################
# 可配置宏变量（改这里即可切新数据）
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 逐行复制调试时，BASH_SOURCE 可能不指向脚本本身；做多级兜底。
DEFAULT_RESMIMIC_ROOT="$SCRIPT_DIR"
if [[ -f "$PWD/source_dev_setup.sh" ]]; then
  DEFAULT_RESMIMIC_ROOT="$PWD"
elif [[ -f "/home/warner/_projects/ResMimic/source_dev_setup.sh" ]]; then
  DEFAULT_RESMIMIC_ROOT="/home/warner/_projects/ResMimic"
fi
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$DEFAULT_RESMIMIC_ROOT}"
GMR_ROOT="${GMR_ROOT:-/home/warner/_projects/GMR}"
CARI4D_ROOT="${CARI4D_ROOT:-/home/warner/_projects/CARI4D}"  # <<< 按需改：CARI4D 仓库根目录
CARI4D_PYTHON="${CARI4D_PYTHON:-/home/warner/miniconda3/envs/cari4d/bin/python}"  # <<< 按需改：用于读取 CARI4D pth
GMR_PYTHON="${GMR_PYTHON:-/home/warner/miniconda3/envs/gmr/bin/python}"  # <<< 按需改：用于 GMR retarget

# 你这次提供的 Step7 输出
CARI4D_PTH="${CARI4D_PTH:-/home/warner/_projects/CARI4D/output/opt/cari4d-release+step031397_demo_20260531-202214-hy3d3-optv2_20260531-202214/Date03_Sub01_bike_May_31_19_34.pth}"  # <<< 改这里
TAG="${TAG:-Date03_Sub01_bike_May_31_19_34}"  # <<< 改这里
SPLIT="${SPLIT:-in}"  # <<< 按需改：in | pr | gt
PAIR_SUFFIX="${PAIR_SUFFIX:-bikez}"  # <<< 改这里
# ===== human + object 总体 xyz 平移 =====
# 总体 xyz 由 runtime pair leveling 自动计算，不手动填 xyz。
# RUNTIME_PAIR_LEVEL_TARGET_Z 控制总体落地后的目标高度。
ENABLE_RUNTIME_PAIR_LEVELING="${ENABLE_RUNTIME_PAIR_LEVELING:-1}"
RUNTIME_PAIR_LEVEL_TARGET_Z="${RUNTIME_PAIR_LEVEL_TARGET_Z:-0.0}"

# ===== human + object 总体 rpy 旋转 =====
# 总体 rpy 同样由 runtime pair leveling 自动计算，不手动填 rpy。
# 下面这个只影响文件级 aligned human motion 是否额外估计整段 yaw。
ALIGN_HUMAN_YAW_TO_OBJECT="${ALIGN_HUMAN_YAW_TO_OBJECT:-0}"

# ===== object 单独 xyz 平移 =====
# 运行时直接加到 object root position；Isaac 和 viser 同时生效。
OBJECT_ROOT_POS_OFFSET_X="${OBJECT_ROOT_POS_OFFSET_X:-0.0}"
OBJECT_ROOT_POS_OFFSET_Y="${OBJECT_ROOT_POS_OFFSET_Y:--0.2}"
OBJECT_ROOT_POS_OFFSET_Z="${OBJECT_ROOT_POS_OFFSET_Z:--0.8}"

# ===== object 单独 rpy 旋转 =====
# ROOT_ROT 是世界坐标系旋转，左乘：q = q_offset * q_motion。
OBJECT_ROOT_ROT_ROLL_DEG="${OBJECT_ROOT_ROT_ROLL_DEG:--90.0}"
OBJECT_ROOT_ROT_PITCH_DEG="${OBJECT_ROOT_ROT_PITCH_DEG:-0.0}"
OBJECT_ROOT_ROT_YAW_DEG="${OBJECT_ROOT_ROT_YAW_DEG:-0.0}"

# LOCAL_ROT 是物体局部坐标系旋转，右乘：q = q_motion * q_offset。
# 先注释掉，后续确认是否需要再放开。
# OBJECT_ROOT_LOCAL_ROT_ROLL_DEG="${OBJECT_ROOT_LOCAL_ROT_ROLL_DEG:--10.0}"
# OBJECT_ROOT_LOCAL_ROT_PITCH_DEG="${OBJECT_ROOT_LOCAL_ROT_PITCH_DEG:-0.0}"
# OBJECT_ROOT_LOCAL_ROT_YAW_DEG="${OBJECT_ROOT_LOCAL_ROT_YAW_DEG:-0.0}"

# ===== human 单独 rpy/高度补偿 =====
# 这组只修机器人 root 姿态和初始高度；Isaac 和 viser 同时生效。
HUMAN_ROOT_ROT_ROLL_DEG="${HUMAN_ROOT_ROT_ROLL_DEG:--90.0}"
HUMAN_ROOT_ROT_PITCH_DEG="${HUMAN_ROOT_ROT_PITCH_DEG:--20.0}"
HUMAN_ROOT_ROT_YAW_DEG="${HUMAN_ROOT_ROT_YAW_DEG:-0.0}"
HUMAN_ROOT_Z_BIAS="${HUMAN_ROOT_Z_BIAS:-0.05}"
OBJECT_ROOT_Z_BIAS="${OBJECT_ROOT_Z_BIAS:-0.03}"

# 任务与训练
TASK="${TASK:-g1_hoi_bike_cari4d}"  # bike 任务名
PROJ_NAME="${PROJ_NAME:-bike}"
EXPTID="${EXPTID:-bike_gui_train}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1}"
WAND_ENTITY="${WAND_ENTITY:-warner0709-shanghai-ai-lab}"
TRAIN_HEADLESS="${TRAIN_HEADLESS:-0}"  # 0=默认有头训练，1=headless训练
TRAIN_MAX_ITERATIONS="${TRAIN_MAX_ITERATIONS:-900}"

# 载入 IsaacGym 后的全局偏移请手动改这里：
#   /home/warner/_projects/ResMimic/legged_gym/legged_gym/envs/g1/g1_hoi_bike_cari4d_config.py
# 脚本不再自动覆盖：
#   motion_global_rot_offset_deg
#   motion_global_pos_offset

# 配置文件与 motion 路径
CFG_FILE="$RESMIMIC_ROOT/legged_gym/legged_gym/envs/g1/g1_hoi_bike_cari4d_config.py"
MOTION_DIR="$RESMIMIC_ROOT/assets/motions"
HUMAN_BASE="$MOTION_DIR/${TAG}_human.pkl"
OBJECT_BASE="$MOTION_DIR/${TAG}_object.npz"
HUMAN_PAIR="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}.pkl"
OBJECT_PAIR="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}.npz"
OBJECT_MESH="$RESMIMIC_ROOT/assets/hy3d_bike_cari4d/Date03_Sub01_bike_wild001_000_align_centered_x20.obj"
PRE_HUMAN="$MOTION_DIR/${TAG}_smplx_input.npz"
ALIGNED_HUMAN="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}_aligned.pkl"
ALIGNED_OBJECT="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}_aligned.npz"
GROUNDED_HUMAN="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}_aligned_grounded.pkl"
GROUNDED_OBJECT="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}_aligned_grounded.npz"

# ===== 逐行调试：手动生成配对文件（可直接复制执行）=====
# cp -f "$HUMAN_BASE" "$HUMAN_PAIR"
# cp -f "$OBJECT_BASE" "$OBJECT_PAIR"
#
# 你当前这组数据的具体命令（可直接复制）：
# cp -f "/home/warner/_projects/ResMimic/assets/motions/Date03_Sub01_bike_on_may_29_21_17_human.pkl" \
#       "/home/warner/_projects/ResMimic/assets/motions/Date03_Sub01_bike_on_may_29_21_17_human_upright_bikez.pkl"
# cp -f "/home/warner/_projects/ResMimic/assets/motions/Date03_Sub01_bike_on_may_29_21_17_object.npz" \
#       "/home/warner/_projects/ResMimic/assets/motions/Date03_Sub01_bike_on_may_29_21_17_object_upright_bikez.npz"

# 0: 严格要求已有配对修正文件；1: 找不到时自动用基础文件兜底复制
ALLOW_FALLBACK_PAIR="${ALLOW_FALLBACK_PAIR:-0}"  # <<< 按需改

# 基础校验（避免在任意目录手敲片段时路径错位）
[[ -f "$RESMIMIC_ROOT/source_dev_setup.sh" ]] || die "未找到 $RESMIMIC_ROOT/source_dev_setup.sh（当前 RESMIMIC_ROOT=$RESMIMIC_ROOT）。逐行调试可先设置 RESMIMIC_ROOT=/home/warner/_projects/ResMimic。"
[[ -f "$RESMIMIC_ROOT/scripts/export_cari4d_intermediate.py" ]] || die "未找到 CARI4D 导出脚本: $RESMIMIC_ROOT/scripts/export_cari4d_intermediate.py"
[[ -f "$RESMIMIC_ROOT/scripts/retarget_smplx_to_resmimic.py" ]] || die "未找到 GMR retarget 脚本: $RESMIMIC_ROOT/scripts/retarget_smplx_to_resmimic.py"
[[ -f "$RESMIMIC_ROOT/scripts/ground_human_object_pair.py" ]] || die "未找到 ground 配对脚本: $RESMIMIC_ROOT/scripts/ground_human_object_pair.py"
[[ -f "$CFG_FILE" ]] || die "未找到 bike 配置文件: $CFG_FILE"
[[ -f "$CARI4D_PTH" ]] || die "输入 pth 不存在: $CARI4D_PTH"
[[ -d "$CARI4D_ROOT" ]] || die "CARI4D_ROOT 不存在: $CARI4D_ROOT"
[[ -x "$CARI4D_PYTHON" ]] || die "CARI4D_PYTHON 不可执行: $CARI4D_PYTHON"
[[ -x "$GMR_PYTHON" ]] || die "GMR_PYTHON 不可执行: $GMR_PYTHON"

echo "[INFO] RESMIMIC_ROOT=$RESMIMIC_ROOT"

########################################
# 1) 环境 + 转换
########################################

source "$RESMIMIC_ROOT/source_dev_setup.sh"
cd "$RESMIMIC_ROOT"

# Align runtime loader path with the actual active Python env so Isaac Gym can
# resolve libpython3.8.so.1.0 from the same env as `python`.
ACTIVE_PY_PREFIX="$(python - <<'PY'
import sys
print(sys.prefix)
PY
)"
if [[ -n "$ACTIVE_PY_PREFIX" && -d "$ACTIVE_PY_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$ACTIVE_PY_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$RESMIMIC_ROOT/legged_gym:${PYTHONPATH:-}"

echo "[1/5] 用 CARI4D 环境从 pth 导出 smplx/object..."
EXPORT_ARGS=(
  --cari4d_pth "$CARI4D_PTH"
  --split "$SPLIT"
  --tag "$TAG"
  --resmimic_root "$RESMIMIC_ROOT"
  --cari4d_root "$CARI4D_ROOT"
  --motion_dir "$MOTION_DIR"
  --pair_suffix "$PAIR_SUFFIX"
)

"$CARI4D_PYTHON" "$RESMIMIC_ROOT/scripts/export_cari4d_intermediate.py" "${EXPORT_ARGS[@]}"

echo "[1.5/5] 用 GMR 环境将 smplx retarget 到 ResMimic human..."
"$GMR_PYTHON" "$RESMIMIC_ROOT/scripts/retarget_smplx_to_resmimic.py" \
  --tag "$TAG" \
  --robot "unitree_g1" \
  --tgt_fps 30 \
  --resmimic_root "$RESMIMIC_ROOT" \
  --gmr_root "$GMR_ROOT" \
  --motion_dir "$MOTION_DIR" \
  --pair_suffix "$PAIR_SUFFIX"

########################################
# 2) 配对文件检查
########################################

echo "[2/5] 检查 bike 配对 motion 文件..."
if [[ -f "$HUMAN_PAIR" && -f "$OBJECT_PAIR" ]]; then
  echo "[OK] 已找到配对文件:"
  echo "  $HUMAN_PAIR"
  echo "  $OBJECT_PAIR"
else
  echo "[WARN] 未找到配对文件:"
  echo "  $HUMAN_PAIR"
  echo "  $OBJECT_PAIR"

  if [[ "$ALLOW_FALLBACK_PAIR" == "1" ]]; then
    echo "[2.5/5] 启用兜底：从基础文件复制为配对命名..."
    [[ -f "$HUMAN_BASE" ]] || die "兜底失败：基础人体文件不存在 $HUMAN_BASE"
    [[ -f "$OBJECT_BASE" ]] || die "兜底失败：基础物体文件不存在 $OBJECT_BASE"
    cp -f "$HUMAN_BASE" "$HUMAN_PAIR"
    cp -f "$OBJECT_BASE" "$OBJECT_PAIR"
  else
    die "当前为严格模式（ALLOW_FALLBACK_PAIR=0），缺少配对文件。请先准备好 *_upright_${PAIR_SUFFIX}，或把 ALLOW_FALLBACK_PAIR 设为 1。"
  fi
fi

########################################
# 3) 比较并刚体对齐 human root
########################################

echo "[3/5] 比较 pre/post GMR root motion，并对齐 human 到 object 水平面..."
[[ -f "$PRE_HUMAN" ]] || die "缺少 pre-GMR SMPL-X 文件: $PRE_HUMAN"
[[ -f "$HUMAN_PAIR" ]] || die "缺少 post-GMR human 文件: $HUMAN_PAIR"
[[ -f "$OBJECT_PAIR" ]] || die "缺少 object 文件: $OBJECT_PAIR"

python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$HUMAN_PAIR" \
  --object "$OBJECT_PAIR"

ALIGN_ARGS=()
if [[ "$ALIGN_HUMAN_YAW_TO_OBJECT" == "1" ]]; then
  ALIGN_ARGS+=(--align-yaw)
fi
python "$RESMIMIC_ROOT/scripts/align_human_root_to_object.py" \
  --human "$HUMAN_PAIR" \
  --object "$OBJECT_PAIR" \
  --output "$ALIGNED_HUMAN" \
  "${ALIGN_ARGS[@]}"

python "$RESMIMIC_ROOT/scripts/compare_gmr_root_motion.py" \
  --pre-human "$PRE_HUMAN" \
  --post-human "$ALIGNED_HUMAN" \
  --object "$OBJECT_PAIR"

echo "[3.5/5] 旧 ground 脚本已停用，改为 Isaac Gym 载入阶段做共享 pair leveling..."
cp -f "$OBJECT_PAIR" "$ALIGNED_OBJECT"
# 旧逻辑保留为注释，便于对照：
# python "$RESMIMIC_ROOT/scripts/ground_human_object_pair.py" \
#   --human "$ALIGNED_HUMAN" \
#   --object "$ALIGNED_OBJECT" \
#   --object-mesh "$OBJECT_MESH" \
#   --output-human "$GROUNDED_HUMAN" \
#   --output-object "$GROUNDED_OBJECT" \
#   --target-ground-z "$TARGET_PAIR_GROUND_Z" \
#   --stat first

########################################
# 4) 自动同步 bike config（含 IsaacGym 载入阶段偏移）
########################################

echo "[4/6] 自动同步 bike config（motion 路径 + root rotation offsets）..."
export CFG_FILE TAG PAIR_SUFFIX
export HUMAN_ROOT_ROT_ROLL_DEG HUMAN_ROOT_ROT_PITCH_DEG HUMAN_ROOT_ROT_YAW_DEG
export OBJECT_ROOT_ROT_ROLL_DEG OBJECT_ROOT_ROT_PITCH_DEG OBJECT_ROOT_ROT_YAW_DEG
# LOCAL_ROT 保留注释（如有需要再手动放开）
# export OBJECT_ROOT_LOCAL_ROT_ROLL_DEG OBJECT_ROOT_LOCAL_ROT_PITCH_DEG OBJECT_ROOT_LOCAL_ROT_YAW_DEG
export OBJECT_ROOT_POS_OFFSET_X OBJECT_ROOT_POS_OFFSET_Y OBJECT_ROOT_POS_OFFSET_Z
export ENABLE_RUNTIME_PAIR_LEVELING RUNTIME_PAIR_LEVEL_TARGET_Z
export HUMAN_ROOT_Z_BIAS OBJECT_ROOT_Z_BIAS
python - <<'PY'
import os
import re

cfg = os.environ["CFG_FILE"]
tag = os.environ["TAG"]
pair = os.environ["PAIR_SUFFIX"]
human_root_rot = f'[{float(os.environ["HUMAN_ROOT_ROT_ROLL_DEG"]):.1f}, {float(os.environ["HUMAN_ROOT_ROT_PITCH_DEG"]):.1f}, {float(os.environ["HUMAN_ROOT_ROT_YAW_DEG"]):.1f}]'
object_root_rot = f'[{float(os.environ["OBJECT_ROOT_ROT_ROLL_DEG"]):.1f}, {float(os.environ["OBJECT_ROOT_ROT_PITCH_DEG"]):.1f}, {float(os.environ["OBJECT_ROOT_ROT_YAW_DEG"]):.1f}]'
object_root_pos_offset = f'[{float(os.environ["OBJECT_ROOT_POS_OFFSET_X"]):.3f}, {float(os.environ["OBJECT_ROOT_POS_OFFSET_Y"]):.3f}, {float(os.environ["OBJECT_ROOT_POS_OFFSET_Z"]):.3f}]'
enable_pair_leveling = "True" if os.environ["ENABLE_RUNTIME_PAIR_LEVELING"] == "1" else "False"
pair_level_target_z = float(os.environ["RUNTIME_PAIR_LEVEL_TARGET_Z"])
human_root_z_bias = float(os.environ["HUMAN_ROOT_Z_BIAS"])
object_root_z_bias = float(os.environ["OBJECT_ROOT_Z_BIAS"])

with open(cfg, "r", encoding="utf-8") as f:
    s = f.read()

motion_line = f'motion_file = f"{{REPO_ROOT_DIR}}/assets/motions/{tag}_human_upright_{pair}_aligned.pkl"'
obj_line = f'object_motion_file = f"{{REPO_ROOT_DIR}}/assets/motions/{tag}_object_upright_{pair}_aligned.npz"'

s = re.sub(r'^\s*motion_file\s*=\s*f"\{REPO_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + motion_line, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_motion_file\s*=\s*f"\{REPO_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + obj_line, s, flags=re.MULTILINE)
s = re.sub(r'^\s*enable_runtime_pair_leveling\s*=\s*(True|False)\s*$', '        enable_runtime_pair_leveling = ' + enable_pair_leveling, s, flags=re.MULTILINE)
s = re.sub(r'^\s*runtime_pair_level_target_z\s*=\s*[-0-9.]+\s*$', f'        runtime_pair_level_target_z = {pair_level_target_z:.3f}', s, flags=re.MULTILINE)
s = re.sub(r'^\s*human_root_z_bias\s*=\s*[-0-9.]+\s*$', f'        human_root_z_bias = {human_root_z_bias:.3f}', s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_root_z_bias\s*=\s*[-0-9.]+\s*$', f'        object_root_z_bias = {object_root_z_bias:.3f}', s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_root_pos_offset\s*=\s*\[.*?\]\s*$', '        object_root_pos_offset = ' + object_root_pos_offset, s, flags=re.MULTILINE)
s = re.sub(r'^\s*human_root_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        human_root_rot_offset_deg = ' + human_root_rot, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_root_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        object_root_rot_offset_deg = ' + object_root_rot, s, flags=re.MULTILINE)
# LOCAL_ROT 本次不再从脚本注入配置（避免 unset 导致的 KeyError），如需再启用请放开上方逻辑并手动复用以下行：
# s = re.sub(r'^\s*object_root_local_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        object_root_local_rot_offset_deg = ' + object_root_local_rot, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_motion_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]', s, flags=re.MULTILINE)

with open(cfg, "w", encoding="utf-8") as f:
    f.write(s)

print("[OK] Config updated:", cfg)
PY

########################################
# 5) 启动非物理可视化
########################################

echo "[5/6] 启动 nonphysics viewer..."
PRE_HUMAN="$PRE_HUMAN" \
POST_HUMAN="$ALIGNED_HUMAN" \
OBJECT_MOTION="$ALIGNED_OBJECT" \
ALIGNED_HUMAN="$ALIGNED_HUMAN" \
ALIGN_HUMAN_YAW_TO_OBJECT="$ALIGN_HUMAN_YAW_TO_OBJECT" \
HUMAN_ROOT_ROT_ROLL_DEG="$HUMAN_ROOT_ROT_ROLL_DEG" \
HUMAN_ROOT_ROT_PITCH_DEG="$HUMAN_ROOT_ROT_PITCH_DEG" \
HUMAN_ROOT_ROT_YAW_DEG="$HUMAN_ROOT_ROT_YAW_DEG" \
OBJECT_ROOT_ROT_ROLL_DEG="$OBJECT_ROOT_ROT_ROLL_DEG" \
OBJECT_ROOT_ROT_PITCH_DEG="$OBJECT_ROOT_ROT_PITCH_DEG" \
OBJECT_ROOT_ROT_YAW_DEG="$OBJECT_ROOT_ROT_YAW_DEG" \
# OBJECT_ROOT_LOCAL_ROT_ROLL_DEG="$OBJECT_ROOT_LOCAL_ROT_ROLL_DEG" \
# OBJECT_ROOT_LOCAL_ROT_PITCH_DEG="$OBJECT_ROOT_LOCAL_ROT_PITCH_DEG" \
# OBJECT_ROOT_LOCAL_ROT_YAW_DEG="$OBJECT_ROOT_LOCAL_ROT_YAW_DEG" \
OBJECT_ROOT_POS_OFFSET_X="$OBJECT_ROOT_POS_OFFSET_X" \
OBJECT_ROOT_POS_OFFSET_Y="$OBJECT_ROOT_POS_OFFSET_Y" \
OBJECT_ROOT_POS_OFFSET_Z="$OBJECT_ROOT_POS_OFFSET_Z" \
ENABLE_RUNTIME_PAIR_LEVELING="$ENABLE_RUNTIME_PAIR_LEVELING" \
RUNTIME_PAIR_LEVEL_TARGET_Z="$RUNTIME_PAIR_LEVEL_TARGET_Z" \
HUMAN_ROOT_Z_BIAS="$HUMAN_ROOT_Z_BIAS" \
OBJECT_ROOT_Z_BIAS="$OBJECT_ROOT_Z_BIAS" \
bash "$RESMIMIC_ROOT/run_nonphysics_chair_viewer.sh" &
VIEWER_PID=$!
echo "[INFO] viewer pid=$VIEWER_PID"

########################################
# 6) 启动 Isaac Gym 训练
########################################

echo "[6/6] 启动 Isaac Gym 训练..."
TRAIN_ARGS=(
  --task "$TASK"
  --proj_name "$PROJ_NAME"
  --exptid "$EXPTID"
  --device "$DEVICE"
  --teacher_exptid None
  --num_envs "$NUM_ENVS"
  --wandb_entity "$WAND_ENTITY"
  --max_iterations "$TRAIN_MAX_ITERATIONS"
  
)
if [[ "$TRAIN_HEADLESS" == "1" ]]; then
  TRAIN_ARGS+=(--headless)
fi
python "$RESMIMIC_ROOT/legged_gym/legged_gym/scripts/train.py" "${TRAIN_ARGS[@]}"
