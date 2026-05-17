# ResMimic: From General Motion Tracking to Humanoid Whole-Body Loco-Manipulation via Residual Learning

**Siheng Zhao**, Yanjie Ze, Yue Wang, C. Karen Liu†, Pieter Abbeel†, Guanya Shi†, Rocky Duan†  
*† Equal Advising · 2025*

[**Website**](https://resmimic.github.io/) · [**arXiv**](https://arxiv.org/abs/2510.05070) · [**Video**](https://www.youtube.com/watch?v=dadOH-TpbRk)


---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation and Contact](#citation-and-contact)
- [Acknowledgments](#acknowledgments)

---

## Installation

One-time setup (creates environment and installs dependencies):

```bash
source setup_dev.sh
```

Each time you open a new terminal, activate the environment:

```bash
source source_dev_setup.sh
```

---

## Usage

### Training

Train the residual policy with the provided config. You need to pass your Weights & Biases entity (username or team):

```bash
./train.sh <wandb_entity>
./train.sh warner0709-shanghai-ai-lab
```

Example: `./train.sh my_username`

Viewer mode (interactive window):

```bash
source source_dev_setup.sh
python legged_gym/legged_gym/scripts/train.py --task "g1_hoi_bike" \
	--proj_name "resmimic_bike" --exptid "bike_top_tube" --device cuda:0 \
	--teacher_exptid None --num_envs 512 --wandb_entity warner0709-shanghai-ai-lab

python legged_gym/legged_gym/scripts/train.py --task "g1_hoi_bike" \
        --proj_name "resmimic_bike" --exptid "bike_top_tube" --device cuda:0 \
        --teacher_exptid None --num_envs 4096 --wandb_entity warner0709-shanghai-ai-lab --headless
```

Note: omit `--headless` to keep the viewer on.

### Evaluation

Run evaluation (play) for a trained run. From the repository root:

```bash
./eval.sh <wandb_run_id> <checkpoint_iter>

./eval.sh ou43tqqk 900 --record_video
```

Example: `./eval.sh abc123def 5000` runs evaluation for run `abc123def` at checkpoint iteration 5000.

Optional: uncomment `--record_video` in `eval.sh` to save videos.

Bike HOI evaluation (checkpoint-based, with video recording):

```bash
source source_dev_setup.sh
cd legged_gym/legged_gym/scripts
python play_residual.py --task "g1_hoi_bike" \
	--proj_name "resmimic_bike" \
        --exptid "bike_top_tube" \
        --checkpoint 200 \
	--num_envs 1\
        --device "cuda:0" \
        --teacher_exptid None \
        --teacher_checkpoint -1 \
	--no_wandb \
        --wandb_entity warner0709-shanghai-ai-lab \
        --record_video
```

---

## Citation and Contact

If you find this work useful, please cite:

```bibtex
@article{zhao2025resmimic,
title={ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning}, 
author={Siheng Zhao and Yanjie Ze and Yue Wang and C. Karen Liu and Pieter Abbeel and Guanya Shi and Rocky Duan},
year={2025},
journal= {arXiv preprint arXiv:2510.05070}
}
```
And also consider citing the related works:
```bibtex
@article{ze2025twist2,
title={TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System},
author= {Yanjie Ze and Siheng Zhao and Weizhuo Wang and Angjoo Kanazawa and Rocky Duan and Pieter Abbeel and Guanya Shi and Jiajun Wuand C. Karen Liu},
year= {2025},
journal= {arXiv preprint arXiv:2511.02832}
}
```

Questions: [sihengz@usc.edu](mailto:sihengz@usc.edu)

---

## Acknowledgments

This codebase is built on [TWIST2](https://github.com/amazon-far/TWIST2).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

This repository builds upon third-party open-source projects, including
[`legged_gym`](https://github.com/leggedrobotics/legged_gym) and
[`rsl_rl`](https://github.com/leggedrobotics/rsl_rl). These components have been modified to support our use case. The original
LICENSE files in their respective subdirectories are retained and continue to
govern those portions of the code. All rights and attribution remain with the
original authors in accordance with their licenses.

