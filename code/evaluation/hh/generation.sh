BASE_MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
MODEL_PATH="/ext_hdd/sjkim/parm/exp/hh/parm"
OUTPUT_DIR="/home/sjkim/parm/code/evaluation/results/hh/parm"
NUM_SAMPLES=100

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
    read -r alpha_help alpha_harm alpha_humor<<< "$pair"

    echo "========================================================"
    echo "Running Generation :: Helpfulness: $alpha_help | Harmlessness: $alpha_harm | Humor: $alpha_humor"
    echo "========================================================"

    # 3. 파이썬 스크립트 실행
    CUDA_VISIBLE_DEVICES=2 python generate_outputs.py \
        --model_parm_both_name_or_path "$MODEL_PATH" \
        --model_base_name_or_path "$BASE_MODEL_PATH" \
        --alpha_helpfulness "$alpha_help" \
        --alpha_harmlessness "$alpha_harm" \
        --alpha_humor "$alpha_humor" \
        --normalize_logit False \
        --output_dir "$OUTPUT_DIR" \
        --num_samples $NUM_SAMPLES \

done

echo "All generation tasks finished!"