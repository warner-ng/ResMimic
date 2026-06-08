WANDB_ENTITY=$1

# Original suitcase training command:
# python legged_gym/legged_gym/scripts/train.py --task "g1_hoi" \
#             --proj_name "resmimic_suitcase" \
#             --exptid "suitcase" \
#             --device cuda:0 \
#             --teacher_exptid None \
#             --num_envs 4096 \
#             --wandb_entity "$WANDB_ENTITY"

python legged_gym/legged_gym/scripts/train.py --task "g1_hoi" \
            --proj_name "resmimic_chair" \
            --exptid "chair" \
            --device cuda:0 \
            --teacher_exptid None \
            --num_envs 1 \
            --wandb_entity "$WANDB_ENTITY" \
            --headless False \
            --max_iterations 900
