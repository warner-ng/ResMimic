WANDB_ENTITY=$1

python legged_gym/legged_gym/scripts/train.py --task "g1_hoi" \
            --proj_name "resmimic_suitcase" \
            --exptid "suitcase" \
            --device cuda:0 \
            --teacher_exptid None \
            --num_envs 4096 \
            --wandb_entity "$WANDB_ENTITY"