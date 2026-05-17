cd legged_gym/legged_gym/scripts
wandb_run_id=${1}
wandb_checkpoint_iter=${2}

# Run the evaluation script
python play_residual.py --task "g1_hoi" \
               --proj_name "resmimic_suitcase" \
               --teacher_exptid "None" \
               --exptid "suitcase" \
               --wandb_entity "warner0709-shanghai-ai-lab" \
               --wandb_run_id ${wandb_run_id} \
               --checkpoint ${wandb_checkpoint_iter} \
               --num_envs 1 \
               --device "cuda:0" \
               "${@:3}"
