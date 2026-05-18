1. what is link_body_list

this is 

```bash
(base) warner@warner-Alienware-x16-R1:~/_projects/ResMimic$ python /home/warner/_projects/ResMimic/scripts/inspect_motion_pkl.py /home/warner/_projects/ResMimic/legged_gym/assets/motions/carry.pkl

motion_path: /home/warner/_projects/ResMimic/legged_gym/assets/motions/carry.pkl

keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
fps: type=<class 'int'>, shape=None
root_pos: type=<class 'numpy.ndarray'>, shape=(275, 3)
root_rot: type=<class 'numpy.ndarray'>, shape=(275, 4)
dof_pos: type=<class 'numpy.ndarray'>, shape=(275, 29)
local_body_pos: type=<class 'numpy.ndarray'>, shape=(275, 38, 3)
link_body_list: type=<class 'list'>, shape=None
link_body_list len: 38
link_body_list sample: ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'left_toe_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link']
root_rot first row: [-0.0163585  -0.05248098 -0.40689274  0.9118204 ]
```

---

and this is

```bash

(base) warner@warner-Alienware-x16-R1:~/_projects/ResMimic$ python /home/warner/_projects/ResMimic/scripts/inspect_motion_pkl.py /home/warner/_projects/ResMimic/assets/motions/carrying_bike_rack_g1.pkl

motion_path: /home/warner/_projects/ResMimic/assets/motions/carrying_bike_rack_g1.pkl

keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
fps: type=<class 'int'>, shape=None
root_pos: type=<class 'numpy.ndarray'>, shape=(146, 3)
root_rot: type=<class 'numpy.ndarray'>, shape=(146, 4)
dof_pos: type=<class 'numpy.ndarray'>, shape=(146, 29)
local_body_pos: type=<class 'numpy.ndarray'>, shape=(146, 9, 3)
link_body_list: type=<class 'list'>, shape=None
link_body_list len: 9
link_body_list sample: ['left_rubber_hand', 'right_rubber_hand', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_link', 'right_knee_link', 'left_elbow_link', 'right_elbow_link', 'head_mocap']
root_rot first row: [ 0.05479112 -0.07971334 -0.9009908   0.4229176 ]

```
---

```bash
(base) warner@warner-Alienware-x16-R1:~/_projects/ResMimic$ python /home/warner/_projects/ResMimic/scripts/inspect_motion_pkl.py /home/warner/_projects/ResMimic/assets/motions/carrying_bike_rack_g1_keybody.pkl
motion_path: /home/warner/_projects/ResMimic/assets/motions/carrying_bike_rack_g1_keybody.pkl
keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
fps: type=<class 'int'>, shape=None
root_pos: type=<class 'numpy.ndarray'>, shape=(146, 3)
root_rot: type=<class 'numpy.ndarray'>, shape=(146, 4)
dof_pos: type=<class 'numpy.ndarray'>, shape=(146, 29)
local_body_pos: type=<class 'numpy.ndarray'>, shape=(146, 9, 3)
link_body_list: type=<class 'list'>, shape=None
link_body_list len: 9
link_body_list sample: ['left_rubber_hand', 'right_rubber_hand', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_knee_link', 'right_knee_link', 'left_elbow_link', 'right_elbow_link', 'head_mocap']
root_rot first row: [ 0.05479112 -0.07971334 -0.9009908   0.4229176 ]
```

---


```bash
(base) warner@warner-Alienware-x16-R1:~/_projects/ResMimic$ python /home/warner/_projects/ResMimic/scripts/inspect_motion_pkl.py assets/motions/carrying_bike_rack_g1_fullbody.pkl
motion_path: assets/motions/carrying_bike_rack_g1_fullbody.pkl
keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
fps: type=<class 'int'>, shape=None
root_pos: type=<class 'numpy.ndarray'>, shape=(146, 3)
root_rot: type=<class 'numpy.ndarray'>, shape=(146, 4)
dof_pos: type=<class 'numpy.ndarray'>, shape=(146, 29)
local_body_pos: type=<class 'numpy.ndarray'>, shape=(146, 38, 3)
link_body_list: type=<class 'list'>, shape=None
link_body_list len: 38
link_body_list sample: ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'left_toe_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link']
root_rot first row: [ 0.05479112 -0.07971334 -0.9009908   0.4229176 ]
```


2. what is the training pipeline
我建议按下面顺序，一次一块：

✅ 训练主线建议顺序
train.py
task_registry.py
g1_hoi.py
g1_hoi_config.py
motion_lib_hoi.py
motion_lib_pkl.py
humanoid_mimic.py
相关 reward / obs 计算
资产加载（object/robot asset）


我们现在来开始读这个代码

现在我们的目标：让resmimic直接适配bikeRacking

我们继续读剩下的文件，读完了这些之后我要知道它图中提到的Object Tracking Reward和contact reward contact force virtual contact 这些关键部位的实现逻辑

- 搬箱子的robot motion文件是kneel.pkl with 

keys: ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
root_pos: (250, 3)
root_rot: (250, 4)
dof_pos: (250, 29)
local_body_pos: (250, 38, 3)
link_body_list: 38 items

- but the object motion is kneel.npz with

keys: ['rot', 'trans']
rot: (250, 4)
trans: (250, 3)


