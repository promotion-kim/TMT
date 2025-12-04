BASE_MODEL_PATH="meta-llama/Llama-2-7b-chat"
MODEL_PATH="/ext_hdd/sjkim/parm/exp/hh/parm"
OUTPUT_DIR="/home/sjkim/parm/code/evaluation/results/hh"
NUM_SAMPLES=1

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

    echo "========================================================"
    echo "Running Generation :: Helpfulness: $alpha_help | Harmlessness: $alpha_harm"
    echo "========================================================"

    # 3. 파이썬 스크립트 실행
    CUDA_VISIBLE_DEVICES=0 python generate_outputs.py \
        --model_parm_both_name_or_path "$MODEL_PATH" \
        --model_base_name_or_path "$BASE_MODEL_PATH" \
        --alpha_helpfulness "$alpha_help" \
        --alpha_harmlessness "$alpha_harm" \
        --normalize_logit False \
        --output_dir "$OUTPUT_DIR" \
        --num_samples $NUM_SAMPLES \

done

echo "All generation tasks finished!"
