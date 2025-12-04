ALGORITHM="stch"

PREFERENCES=(
    "0.0 0.0 1.0"
    "0.0 0.1 0.9"
    "0.0 0.2 0.8"
    "0.0 0.3 0.7"
    "0.0 0.4 0.6"
    "0.0 0.5 0.5"
    "0.0 0.6 0.4"
    "0.0 0.7 0.3"
    "0.0 0.8 0.2"
    "0.0 0.9 0.1"
    "0.0 1.0 0.0"
    "0.1 0.0 0.9"
    "0.2 0.0 0.8"
    "0.3 0.0 0.7"
    "0.4 0.0 0.6"
    "0.5 0.0 0.5"
    "0.6 0.0 0.4"
    "0.7 0.0 0.3"
    "0.8 0.0 0.2"
    "0.9 0.0 0.1"
    "1.0 0.0 0.0"
    "0.1 0.9 0.0"
    "0.2 0.8 0.0"
    "0.3 0.7 0.0"
    "0.4 0.6 0.0"
    "0.5 0.5 0.0"
    "0.6 0.4 0.0"
    "0.7 0.3 0.0"
    "0.8 0.2 0.0"
    "0.9 0.1 0.0"
    "0.1 0.1 0.8"
    "0.1 0.8 0.1"
    "0.2 0.2 0.6"
    "0.2 0.4 0.4"
    "0.2 0.6 0.2"
    "0.33 0.33 0.33"
    "0.4 0.4 0.2"
    "0.4 0.2 0.4" 
    "0.6 0.2 0.2"
    "0.8 0.1 0.1"
)

for pair in "${PREFERENCES[@]}"; do
    read -r alpha_help alpha_harm alpha_humor <<< "$pair"
    MODEL_PATH="/home/sjkim/parm/code/evaluation/results/hh/${ALGORITHM}/PARM_${alpha_help}help_${alpha_harm}harm_${alpha_humor}humor"

    echo "========================================================"
    echo "Running Reward Computation :: Help: $alpha_help | Harm: $alpha_harm | Humor: $alpha_humor"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=1 python compute_reward.py \
        --path "$MODEL_PATH" \

done

echo "All computing reward tasks finished!"