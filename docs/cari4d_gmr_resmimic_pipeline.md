# CARI4D Chair → ResMimic（g1_hoi_cari4d_chair）全链路复盘与操作手册

本文是我们**为了解决 chair 数据接入 ResMimic**而编写的实操文档，记录本次已跑通的数据链路、关键配置、常见坑和修复方式。目标是让后续同学可以按本文一次走通，不再重复踩坑。

---

## 1. 目标与范围

- 任务：`g1_hoi_cari4d_chair`
- 目标：
  1. 将 CARI4D 的人体/物体序列转换为 ResMimic 可用的 HOI 运动文件。
  2. 保证**物体空间关系不被破坏**（人-物相对位姿一致）。
  3. 解决初始高度不合理问题（参考 suitcase/carry 任务基线）。
  4. 在可视化（有头）模式下跑通训练链路。

---

## 2. 已跑通的数据链路（高层）

1. 从 CARI4D 输出 `.pth` 读取人体（SMPL）与物体（`pose_abs`）时序。
2. 生成 GMR 可读的 SMPL-X 输入（`*_smplx_input.npz`）。
3. 通过 GMR retarget 到机器人，导出人体轨迹 `*_human.pkl`。
4. 导出物体轨迹 `*_object.npz`（`trans` + `rot`，`rot` 为 `xyzw`）。
5. 对人体与物体做**配对一致**的姿态/高度修正，得到：
  - `*_human_upright_chairz.pkl`
  - `*_object_upright_chairz.npz`
  - 说明：这里的 `chairz` 用于标记“chair 任务的 z 基线对齐版本”。
6. 在任务配置里切换到这对文件，并将额外旋转偏移清零。
7. 用 viewer 进行 smoke test（有头模式），确认物体摆放与朝向正常。
8. 启动训练（有头），链路可进入训练流程。

---

## 3. 按时间顺序的全部代码修改 + 全部操作命令（含关键原理）

### T0：目标约束确认

- 任务固定：`g1_hoi_cari4d_chair`
- 原则：不乱改算法主体，优先通过**数据与路径**修复 chair 放置问题。

### T1：定位与新增的代码/脚本

1. 新增转换脚本：
   - `scripts/prepare_cari4d_hoi_motion.py`
   - 功能：读取 CARI4D `.pth`，生成
     - `*_smplx_input.npz`
     - `*_human.pkl`
     - `*_object.npz`

2. 任务配置文件（chair）更新：
   - `legged_gym/legged_gym/envs/g1/g1_hoi_cari4d_chair_config.py`
   - 关键改动：
     - `object_motion_rot_offset_deg = [0.0, 0.0, 0.0]`
     - `motion_file` 和 `object_motion_file` 指向同一批配对修正文件
     - DAgger 配置类同步同样设置

3. 训练稳定性补丁（wandb）：
   - `legged_gym/legged_gym/scripts/train.py`
   - `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`
   - 目的：防止 `wandb.init` 失败后仍 `wandb.log/save` 崩溃

### T2：最关键步骤——“人/物配对一致”的姿态与高度修正是怎么做的

设每一帧：

- 人体根位姿为 $T_h(t)$
- 物体位姿为 $T_o(t)$
- 人-物相对位姿为
  $$T_{rel}(t)=T_h(t)^{-1}T_o(t)$$

修正时我们不是只改人或只改物，而是对两者施加**同一个外部刚体变换** $C$（包含 upright 旋转与 z 基线平移）：

- $T'_h(t)=C\,T_h(t)$
- $T'_o(t)=C\,T_o(t)$

于是：

$$
T'_{rel}(t)=T'_h(t)^{-1}T'_o(t)
=(C T_h)^{-1}(C T_o)
=T_h^{-1}T_o
=T_{rel}(t)
$$

这就保证了“看起来摆正了”的同时，**人和物体之间的原始空间关系不漂移**。

本次产物命名为：

- `*_human_upright_chairz.pkl`
- `*_object_upright_chairz.npz`

其中 `chairz` 仅表示 chair 任务下的 z 基线对齐版本。

### T3：按时间顺序执行过的命令（原样记录）

1. 初次直接训练（失败，因环境/上下文未完全按 README 进入）：

```bash
cd /home/warner/_projects/ResMimic/legged_gym && python legged_gym/scripts/train.py --task g1_hoi_cari4d_chair --proj_name chair --exptid chair_gui_train --no_wandb
```

2. 采用 README 方式进入环境（正确）：

```bash
source /home/warner/_projects/ResMimic/source_dev_setup.sh
cd /home/warner/_projects/ResMimic
python legged_gym/legged_gym/scripts/train.py \
  --task "g1_hoi_cari4d_chair" \
  --proj_name "chair" \
  --exptid "chair_gui_train" \
  --device cuda:0 \
  --teacher_exptid None \
  --num_envs 1 \
  --no_wandb \
  --wandb_entity warner0709-shanghai-ai-lab
```

3. 退出码说明：
   - `Exit Code: 130` 为人工中断（通常 `Ctrl+C`），不是训练链路本身崩溃。

### T4：本链路实际使用/生成的关键文件

- 配置：
  - `legged_gym/legged_gym/envs/g1/g1_hoi_cari4d_chair_config.py`
- 资产：
  - `legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_smplx_input.npz`
  - `legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_human_upright_chairz.pkl`
  - `legged_gym/assets/motions/Date03_Sub03_chairblack_lift_it1_input_object_upright_chairz.npz`
- 脚本：
  - `scripts/prepare_cari4d_hoi_motion.py`

---

## 5. 中间踩到的坑与修复

### 坑 1：人/物文件“混搭”导致空间关系错位

**现象**
- 人看起来立起来了，但物体位置/朝向不对，或跟人关系漂移。

**根因**
- 使用了 `human_upright.pkl + original_object.npz` 这类非同源配对，破坏人-物相对位姿。

**修复**
- 人体和物体必须作为一对同步修正并同时切换路径：
  - `human_upright_chairz.pkl`
  - `object_upright_chairz.npz`
  - 其中 `chairz` 表示 chair 任务下的对齐版本，必须人/物成对使用。

---

### 坑 2：初始高度偏高

**现象**
- 一开始整体“悬”得高，不像 suitcase/carry 基线。

**根因**
- 仅修正朝向，未把 z 基线与参考任务对齐。

**修复**
- 在配对修正阶段同步调整人体与物体的 z 基线（同一变换作用到两者）。

---

### 坑 3：重复旋转补偿（双重旋转）

**现象**
- 明明文件已修正，运行后又被“拧”了一次。

**根因**
- 文件内已经补偿了姿态，但 config 里的 `object_motion_rot_offset_deg` 仍非零。

**修复**
- 对已修正文件，配置里将 `object_motion_rot_offset_deg = [0,0,0]`。

---

### 坑 4：`wandb.init()` 失败后仍 `wandb.log()` 崩溃

**现象**
- 报错：`You must call wandb.init() before wandb.log()`。

**根因**
- no-wandb/初始化失败路径下，没有对 `wandb.log/save` 做 run 存在性保护。

**修复（已做）**
- 在训练入口和 runner 增加保护：
  - 初始化失败时回退 disabled 模式；
  - 无 active run 时跳过 `wandb.log/save`。

涉及文件：
- `legged_gym/legged_gym/scripts/train.py`
- `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py`
