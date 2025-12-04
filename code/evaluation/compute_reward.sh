ALGORITHM="stch"

PREFERENCES=(
    "0.0 1.0"
    "0.1 0.9"
    "0.2 0.8"
    "0.3 0.7"
    "0.4 0.6"
    "0.5 0.5"
    "0.6 0.4"
    "0.7 0.3"
    "0.8 0.2"
    "0.9 0.1"
    "1.0 0.0"
)

for pair in "${PREFERENCES[@]}"; do
    read -r alpha_help alpha_harm <<< "$pair"
    MODEL_PATH="/home/sjkim/parm/code/evaluation/results/${ALGORITHM}/PARM_${alpha_help}help_${alpha_harm}harm"

    echo "========================================================"
    echo "Running Reward Computation :: Help: $alpha_help | Harm: $alpha_harm"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=2 python compute_reward.py \
        --path "$MODEL_PATH" \

done

echo "All computing reward tasks finished!"