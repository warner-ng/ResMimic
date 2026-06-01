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

# 你这次提供的 Step7 输出
CARI4D_PTH="${CARI4D_PTH:-/home/warner/_projects/CARI4D/output/opt/cari4d-release+step031397_demo_oomfix-hy3d3-optv2_oomfix/Date03_Sub01_bike_on_may_29_21_17.pth}"  # <<< 改这里
TAG="${TAG:-Date03_Sub01_bike_on_may_29_21_17}"  # <<< 改这里
SPLIT="${SPLIT:-in}"  # <<< 按需改：in | pr | gt
PAIR_SUFFIX="${PAIR_SUFFIX:-bikez}"  # <<< 改这里

# 任务与训练
TASK="${TASK:-g1_hoi_bike_cari4d}"  # bike 任务名
PROJ_NAME="${PROJ_NAME:-bike}"
EXPTID="${EXPTID:-bike_gui_train}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1}"
WAND_ENTITY="${WAND_ENTITY:-warner0709-shanghai-ai-lab}"

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
[[ -f "$RESMIMIC_ROOT/scripts/prepare_cari4d_hoi_motion.py" ]] || die "未找到转换脚本: $RESMIMIC_ROOT/scripts/prepare_cari4d_hoi_motion.py"
[[ -f "$CFG_FILE" ]] || die "未找到 bike 配置文件: $CFG_FILE"
[[ -f "$CARI4D_PTH" ]] || die "输入 pth 不存在: $CARI4D_PTH"
[[ -d "$CARI4D_ROOT" ]] || die "CARI4D_ROOT 不存在: $CARI4D_ROOT"

echo "[INFO] RESMIMIC_ROOT=$RESMIMIC_ROOT"

########################################
# 1) 环境 + 转换
########################################

source "$RESMIMIC_ROOT/source_dev_setup.sh"
cd "$RESMIMIC_ROOT"

echo "[1/5] 从 CARI4D pth 生成 smplx/human/object..."
python "$RESMIMIC_ROOT/scripts/prepare_cari4d_hoi_motion.py" \
  --cari4d_pth "$CARI4D_PTH" \
  --split "$SPLIT" \
  --tag "$TAG" \
  --resmimic_root "$RESMIMIC_ROOT" \
  --gmr_root "$GMR_ROOT" \
  --cari4d_root "$CARI4D_ROOT" \
  --motion_dir "$MOTION_DIR"

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
# 3) 自动同步 bike config（含 IsaacGym 载入阶段偏移）
########################################

echo "[3/5] 自动同步 bike config（仅 motion 路径；load偏移请手动改配置）..."
export CFG_FILE TAG PAIR_SUFFIX
python - <<'PY'
import os
import re

cfg = os.environ["CFG_FILE"]
tag = os.environ["TAG"]
pair = os.environ["PAIR_SUFFIX"]

with open(cfg, "r", encoding="utf-8") as f:
    s = f.read()

motion_line = f'motion_file = f"{{REPO_ROOT_DIR}}/assets/motions/{tag}_human_upright_{pair}.pkl"'
obj_line = f'object_motion_file = f"{{REPO_ROOT_DIR}}/assets/motions/{tag}_object_upright_{pair}.npz"'

s = re.sub(r'^\s*motion_file\s*=\s*f"\{REPO_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + motion_line, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_motion_file\s*=\s*f"\{REPO_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + obj_line, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_motion_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]', s, flags=re.MULTILINE)

with open(cfg, "w", encoding="utf-8") as f:
    f.write(s)

print("[OK] Config updated:", cfg)
for line in s.splitlines():
  t = line.strip()
  if t.startswith("motion_global_rot_offset_deg") or t.startswith("motion_global_pos_offset"):
    print("[MANUAL]", t)
PY

########################################
# 4) 启动训练
########################################

echo "[4/5] 启动训练..."
python "$RESMIMIC_ROOT/legged_gym/legged_gym/scripts/train.py" \
  --task "$TASK" \
  --proj_name "$PROJ_NAME" \
  --exptid "$EXPTID" \
  --device "$DEVICE" \
  --teacher_exptid None \
  --num_envs "$NUM_ENVS" \
  --no_wandb \
  --wandb_entity "$WAND_ENTITY"
