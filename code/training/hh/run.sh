cuda=0,1,2,3
exp_name=assistant_parm_4r1_4r2_1epoch_1.0p

lora_r2=4
lora_r=4
lora_alpha=8

pref_sample_p=1.0

safe_obj=true
help_obj=true
humor_obj=true

beta_safe=1e-2
beta_help=1e-2
beta_humor=1e-2

epoch=1
beta=0.001
learning_rate=5e-4 
bs=32 
per_device_train_batch_size=4 #4

model_name_or_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0'

#!
export WANDB_PROJECT="parm"

###### the following is automatically set
num_GPU=$(echo $cuda | awk -F, '{print NF}')
gradient_accumulation_steps=$(($bs/$num_GPU/$per_device_train_batch_size))
preference_dataset=hh_rlhf
# exp_name=$peft_type-$version-epoch_$epoch

output_dir=/ext_hdd/sjkim/parm/exp/hh/parm
if [ -d "${output_dir}" ]; then
    echo -e "\n\n"
    echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir." >&2
    exit 1
fi
echo "Output dir: $output_dir"

# cd /path/code/training
accelerate launch --gpu_ids $cuda --main_process_port 29500 --num_processes $num_GPU train_pref_arm.py \
    --preference_dataset=$preference_dataset \
    --pref_sample_p=$pref_sample_p \
    --lora_r=$lora_r \
    --lora_r2=$lora_r2 \
    --lora_alpha=$lora_alpha \
    --safe_obj=$safe_obj \
    --help_obj=$help_obj \
    --humor_obj=$humor_obj \
    --beta_safe=$beta_safe \
    --beta_help=$beta_help \
    --beta_humor=$beta_humor \
    --model_name_or_path=$model_name_or_path \
    --beta=$beta \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epoch \
    --output_dir=$output_dir \
    --run_name=$exp_name \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --per_device_eval_batch_size=2 \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --eval_steps=20 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=20 \
    --weight_decay=0.05 \
    --gradient_checkpointing=True \
    --bf16=True \
    --max_prompt_length=512 \
    --max_length=1024 \
    --report_to="wandb" \
    --remove_unused_columns=False 

echo "Finished training $output_dir"