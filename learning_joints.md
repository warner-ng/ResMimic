# Learning Guide: Joint/Body Order, `link_body_list`, and Isaac Gym Rigid Body Indices

This guide explains how I identified the `link_body_list` ordering and matched it to Isaac Gym’s rigid body order, plus how you can independently verify and learn it.

## 1) What `link_body_list` is and why it matters
In ResMimic/GMR motion files, `link_body_list` defines the **ordered list of body names** that correspond to the third dimension of `local_body_pos`:

- `local_body_pos` shape is `(T, N, 3)`
- `link_body_list` length is `N`
- The `i`-th body position in `local_body_pos[:, i, :]` belongs to the body named `link_body_list[i]`

This ordering must match the simulation’s rigid-body order (Isaac Gym) when you compare or broadcast positions during training.

## 2) How I confirmed the order inside `carry.pkl`
I directly inspected the motion file:

- Keys: `fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, `link_body_list`
- Shapes:
  - `root_pos`: `(T, 3)`
  - `root_rot`: `(T, 4)` (quaternion xyzw)
  - `dof_pos`: `(T, 29)`
  - `local_body_pos`: `(T, 38, 3)`
  - `link_body_list`: length `38`

That tells us the file already carries its own body ordering in `link_body_list`.

## 3) How I matched Isaac Gym’s rigid body order
Isaac Gym exposes the rigid body order through the asset. I loaded the G1 URDF with Isaac Gym and queried:

- `gym.get_asset_rigid_body_names(asset)`

This returns the rigid body list **in the exact order** that Gym uses internally (the same order that `rigid_body_states[:, body_id, :]` follows).

For G1 (29-DoF), the list has 38 bodies. The first few are:

```
['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', ...]
```

This matches the order stored in the `carry.pkl` `link_body_list`, confirming the order consistency.

## 4) The method I used (step-by-step)
1. **Inspect the PKL motion file** to see keys, shapes, and `link_body_list`.
2. **Load the robot URDF** in Isaac Gym and query rigid body names.
3. **Compare the two lists** to verify ordering and length match.
4. If motion only provides key bodies (e.g., 9 bodies), **expand it** into full-body order by placing each key body into the correct index and filling the rest with zeros.

加载 g1_custom_collision_29dof.urdf
调用 Isaac Gym API：
这个函数返回的列表就是 Isaac Gym 内部 rigid body 的顺序（即 rigid_body_states[:, body_id, :] 的 body_id 顺序）。

所以“G1 有 38 个 body，前几个是 pelvis / left_hip_pitch_link …”就是这个 API 返回的结果。

## 5) How you can reproduce the checks
### 5.1 Inspect a motion file
- Open the PKL and print keys, shapes, and `link_body_list`.

### 5.2 Get Isaac Gym body order
- Load the URDF with Isaac Gym and print `gym.get_asset_rigid_body_names(asset)`.

### 5.3 Verify consistency
- Ensure the `link_body_list` matches the Gym list exactly in order.

## 6) Practical rules of thumb
- **Always trust `link_body_list` inside the motion file** (if present).
- **Always trust `gym.get_asset_rigid_body_names`** for the simulator’s order.
- If they don’t match, **either reorder `local_body_pos`** or rebuild the motion file.

## 7) Common pitfalls
- Mixing **URDF link order** with **Gym’s rigid body order** — they are not always identical.
- Using only key bodies without expanding to full rigid body order when the training code expects full-body tensors.
- Quaternions are typically **xyzw** in these files (not wxyz).

## 8) Recommended reading (in-repo)
- `pose/pose/utils/motion_lib_pkl.py`
- `pose/pose/utils/motion_lib_hoi.py`
- `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
- `legged_gym/legged_gym/envs/g1/g1_hoi.py`

## 9) Why this matters for training stability
Training computes:

- `key_body_pos = rigid_body_states[:, body_ids, :3]`

If the motion file uses a different ordering than Gym, you’ll see shape or semantic mismatches that lead to large errors or crashes.

---
