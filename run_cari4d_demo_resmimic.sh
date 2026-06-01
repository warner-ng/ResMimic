#!/usr/bin/env bash
set -euo pipefail

# CARI4D Chair -> ResMimic (g1_hoi_cari4d_chair)
# 对应 docs/cari4d_gmr_resmimic_pipeline.md 的“第3节”流程命令汇总（含代码修改动作）
#
# 用法：只改下面“可配置宏变量”即可复用到新数据。

########################################
# 可配置宏变量（改这里即可切新数据）
########################################
#
# ===== 你主要改这几行（按优先级）=====
# 1) CARI4D_PTH
# 2) TAG
# 3) PAIR_SUFFIX（通常 suitcasez / chairz / bikez）
# 其余变量一般不用改，除非你要改任务名、显卡、实验名等。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESMIMIC_ROOT="${RESMIMIC_ROOT:-$SCRIPT_DIR}"
GMR_ROOT="${GMR_ROOT:-/home/warner/_projects/GMR}"

# CARI4D 输入（核心变量）
TAG="${TAG:-Date03_Sub03_chairblack_lift_it1_input}"  # <<< 改这里：该数据的统一tag（文件名前缀）
CARI4D_PTH="${CARI4D_PTH:-/home/warner/_projects/CARI4D/output/coconet/cari4d-release+step031397_demo/Date03_Sub03_chairblack_lift.pth}"  # <<< 改这里：输入xxx.pth
SPLIT="${SPLIT:-in}"  # <<< 按需改：in | pr | gt

# 任务与训练
TASK="${TASK:-g1_hoi_cari4d_chair}"  # <<< 按需改：任务名
PROJ_NAME="${PROJ_NAME:-chair}"       # <<< 按需改：项目名
EXPTID="${EXPTID:-chair_gui_train}"   # <<< 按需改：实验ID
DEVICE="${DEVICE:-cuda:0}"            # <<< 按需改：例如 cuda:1
NUM_ENVS="${NUM_ENVS:-1}"             # <<< 按需改：并行环境数
WAND_ENTITY="${WAND_ENTITY:-warner0709-shanghai-ai-lab}"

# motion 命名（配对一致版本）
PAIR_SUFFIX="${PAIR_SUFFIX:-suitcasez}"  # <<< 改这里：配对后缀（suitcasez/chairz/bikez...）
MOTION_DIR="$RESMIMIC_ROOT/legged_gym/assets/motions"
HUMAN_BASE="$MOTION_DIR/${TAG}_human.pkl"
OBJECT_BASE="$MOTION_DIR/${TAG}_object.npz"
HUMAN_PAIR="$MOTION_DIR/${TAG}_human_upright_${PAIR_SUFFIX}.pkl"
OBJECT_PAIR="$MOTION_DIR/${TAG}_object_upright_${PAIR_SUFFIX}.npz"

# 自动改配置（把第3节“代码修改”也写进脚本）
CFG_FILE="$RESMIMIC_ROOT/legged_gym/legged_gym/envs/g1/g1_hoi_cari4d_chair_config.py"

# 若没有 *_upright_${PAIR_SUFFIX}，是否用基础 human/object 兜底复制一份同名文件
# 0: 严格要求已有配对修正文件；1: 自动兜底（便于快速跑通）
ALLOW_FALLBACK_PAIR="${ALLOW_FALLBACK_PAIR:-0}"  # <<< 按需改：1=找不到配对文件时自动兜底复制

########################################
# T0/T1/T3：环境、转换、训练命令
########################################

source "$RESMIMIC_ROOT/source_dev_setup.sh"
cd "$RESMIMIC_ROOT"

echo "[1/5] 从 CARI4D pth 生成 smplx/human/object..."
python "$RESMIMIC_ROOT/scripts/prepare_cari4d_hoi_motion.py" \
  --cari4d_pth "$CARI4D_PTH" \
  --split "$SPLIT" \
  --tag "$TAG" \
  --resmimic_root "$RESMIMIC_ROOT" \
  --gmr_root "$GMR_ROOT"

########################################
# T2：人/物配对一致文件检查（默认 suitcasez 对）
########################################

echo "[2/5] 检查配对 motion 文件..."
if [[ -f "$HUMAN_PAIR" && -f "$OBJECT_PAIR" ]]; then
  echo "[OK] 已找到配对文件:"
  echo "  $HUMAN_PAIR"
  echo "  $OBJECT_PAIR"
else
  echo "[WARN] 未找到配对文件:"
  echo "  $HUMAN_PAIR"
  echo "  $OBJECT_PAIR"

  if [[ "$ALLOW_FALLBACK_PAIR" == "1" ]]; then
    echo "[3/5] 启用兜底：从基础文件复制为配对命名..."
    cp -f "$HUMAN_BASE" "$HUMAN_PAIR"
    cp -f "$OBJECT_BASE" "$OBJECT_PAIR"
  else
    echo "[STOP] 当前为严格模式（ALLOW_FALLBACK_PAIR=0），请先准备好 *_upright_${PAIR_SUFFIX} 配对文件。"
    exit 1
  fi
fi

########################################
# T1：代码修改动作自动化（config 同步）
########################################

echo "[4/5] 自动同步 chair config 中 motion 路径与旋转偏移..."
export CFG_FILE TAG PAIR_SUFFIX
python - <<'PY'
import os
import re

cfg = os.environ["CFG_FILE"]
tag = os.environ["TAG"]
pair = os.environ["PAIR_SUFFIX"]

with open(cfg, "r", encoding="utf-8") as f:
    s = f.read()

motion_line = f'motion_file = f"{{LEGGED_GYM_ROOT_DIR}}/assets/motions/{tag}_human_upright_{pair}.pkl"'
obj_line = f'object_motion_file = f"{{LEGGED_GYM_ROOT_DIR}}/assets/motions/{tag}_object_upright_{pair}.npz"'

# 同步两处 class motion(...) 里的路径
s = re.sub(r'^\s*motion_file\s*=\s*f"\{LEGGED_GYM_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + motion_line, s, flags=re.MULTILINE)
s = re.sub(r'^\s*object_motion_file\s*=\s*f"\{LEGGED_GYM_ROOT_DIR\}/assets/motions/.*?"\s*$', '        ' + obj_line, s, flags=re.MULTILINE)

# 对已修正文件，配置中偏移固定为 0
s = re.sub(r'^\s*object_motion_rot_offset_deg\s*=\s*\[.*?\]\s*$', '        object_motion_rot_offset_deg = [0.0, 0.0, 0.0]', s, flags=re.MULTILINE)

with open(cfg, "w", encoding="utf-8") as f:
    f.write(s)

print("[OK] Config updated:", cfg)
PY

########################################
# T3：训练命令（文档已验证）
########################################

echo "[5/5] 启动训练..."
python "$RESMIMIC_ROOT/legged_gym/legged_gym/scripts/train.py" \
  --task "$TASK" \
  --proj_name "$PROJ_NAME" \
  --exptid "$EXPTID" \
  --device "$DEVICE" \
  --teacher_exptid None \
  --num_envs "$NUM_ENVS" \
  --no_wandb \
  --wandb_entity "$WAND_ENTITY" \
  --disable_termination_for_debug

# 备注：文档中“早期尝试”命令（保留）
# cd "$RESMIMIC_ROOT/legged_gym" && python legged_gym/scripts/train.py --task g1_hoi_cari4d_chair --proj_name chair --exptid chair_gui_train --no_wandb
