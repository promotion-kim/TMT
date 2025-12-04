'''
adapted from trl/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

Due to data processing steps (data format and model template), this script only supports training LLaMA-1 on the HH-RLHF dataset and training Alpaca on the PKU-SafeRLHF dataset.
For other dataset and models, we need to modify the data processing steps.
'''
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import PBLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig
from trl.commands.cli_utils import TrlParser

from pref_arm_trainer import PrefARMTrainer

import wandb
#wandb.init(mode="disabled")

@dataclass
class ARMConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.0,
        metadata={"help": "target reward margin gamma. The reward margin is reward_win - reward_lose - gamma, where \
                                                          reward_win (reward_lose) = beta * log pi_arm_win (lose)."},
    )
    length_normalization: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize the logprobs using the length of the response. length_normalization=True is not default for ARM and should only be used for testing purposes!"},
    )

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO/ARM training script.

    NOTE: other training arguments, such as learning rate and beta, should be set in the command line.
    They are included in DPOConfig, not here. ScriptArguments below are arguments that are not included in DPOConfig.
    """
    # used by DPO or ARM
    # algorithm: Optional[str] = field(default="arm", metadata={"help": "algorithm to use [dpo, arm]"})
    model_name_or_path: Optional[str] = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "the location of the to-be-finetuned model name or path"},
    )

    # dataset
    preference_dataset: Optional[str] = field(
        default="hh_rlhf", metadata={"help": ""},
    )

    # training
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    # model
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_r2: Optional[int] = field(default=8, metadata={"help": "for mixlora only"})

    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # others
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    # tasks
    safe_obj: Optional[bool] = field(default=True, metadata={"help": ""})
    help_obj: Optional[bool] = field(default=True, metadata={"help": ""})
    humor_obj: Optional[bool] = field(default=True, metadata={"help": ""})
    # beta in loss
    beta_safe: Optional[float] = field(default=0.5, metadata={"help": ""})
    beta_help: Optional[float] = field(default=0.5, metadata={"help": ""})
    beta_humor: Optional[float] = field(default=0.5, metadata={"help": ""})
    #
    pref_sample_p: Optional[float] = field(default=1.0, metadata={"help": ""})

def _parse_dialogue(dialogue_string: str) -> list[dict]:
    """
    Anthropic 스타일의 "\n\nHuman: ... \n\nAssistant: ..." 문자열을
    Hugging Face의 messages 리스트 형식으로 파싱합니다.
    """
    messages = []
    # 데이터는 항상 Human: 턴으로 시작하고 Assistant: 턴으로 끝남 (Assistant: 뒤는 비어있음)
    
    # 맨 앞의 \n\n을 제거합니다.
    dialogue_string = dialogue_string.strip() 

    # 'Assistant:'를 기준으로 문자열을 나눕니다.
    # 이렇게 하면 각 턴이 [Human: Q] [Assistant: A] 쌍으로 나뉘기 쉽습니다.
    
    # 1. 마지막 Assistant: 태그를 제거하여 응답 직전까지의 대화 기록만 남깁니다.
    if dialogue_string.endswith("\n\nAssistant:"):
        dialogue_string = dialogue_string[:-len("\n\nAssistant:")].strip()
    
    # 2. 대화 기록을 턴(Turn)별로 분리합니다.
    # 첫 턴은 Human이므로, Human:을 기준으로 나눕니다.
    turns = dialogue_string.split("\n\nHuman:") 
    
    # 첫 요소는 빈 문자열이거나 시작 부분의 잔여물일 수 있으므로 제거
    turns = [t.strip() for t in turns if t.strip()]

    for full_turn in turns:
        if "\n\nAssistant:" in full_turn:
            # 중간 턴: "질문 \n\nAssistant: 답변" 형태
            try:
                user_content, assistant_content = full_turn.split("\n\nAssistant:", 1)
                messages.append({"role": "user", "content": user_content.strip()})
                messages.append({"role": "assistant", "content": assistant_content.strip()})
            except ValueError:
                # 예상치 못한 분할 (오류 처리)
                print(f"Warning: Could not split turn cleanly: {full_turn[:50]}...")
                continue
        else:
            # 마지막 턴: "최종 질문" 형태 (마지막 Assistant: 태그는 이미 제거되었음)
            messages.append({"role": "user", "content": full_turn.strip()})
    return messages

def get_hh_rlhf(
    dataset_name: str,
    sanity_check: bool = False,
    num_proc=4,
    obj_key=None,
    tokenizer=None # 👈 토크나이저를 인자로 받아야 apply_chat_template 사용 가능
):
    train_dataset = load_dataset("json", data_files='../../data/hh/train.json', 
            split='train', num_proc=num_proc)
    test_dataset = load_dataset("json", data_files='../../data/hh/dev.json', 
            split='train', num_proc=num_proc)
    original_columns = train_dataset.column_names

    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))
    
    # --------------------------------------------------------------------------------------------------
    # 기존 PKU/Chat Template 함수 제거 (이미 위에서 _parse_dialogue로 대체되었음)
    # --------------------------------------------------------------------------------------------------

    def return_prompt_and_responses(sample, obj_key, tokenizer) -> Dict[str, str]:
        
        labels = {'safe': sample['safer_response_id'], 
                  "help": sample['better_response_id'],
                  "humor": sample['humorer_responses_id'],}
        
        # 1. 멀티턴 대화 기록을 메시지 리스트로 파싱
        messages = _parse_dialogue(sample["prompt"])
        SYSTEM_PROMPT = "You are a helpful, harmless, and humorous assistant."
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # 2. Chosen/Rejected 응답 포맷팅
        # chosen/rejected 응답은 'messages' 리스트에 마지막 턴의 응답으로 추가됩니다.
        
        # A. 프롬프트 (Context + Final Instruction) 생성
        # add_generation_prompt=True는 마지막 <|assistant|> 태그를 추가합니다.
        # chosen/rejected 응답을 제외한 대화 기록만 템플릿에 적용합니다.
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True # <|assistant|> 태그를 끝에 추가
        )

        # B. Chosen 응답과 Rejected 응답에 템플릿 적용
        # LLaMA-2/TinyLLaMA 계열은 응답 끝에 </s>를 붙입니다.
        # chosen/rejected는 *이미* 순수한 텍스트 응답이므로,
        # 템플릿 대신, </s> 토큰만 붙여주거나 (혹은 붙이지 않거나, DPO 트레이너에 따라 다름),
        # 안전하게는 모델이 학습된 방식대로 턴을 포함해야 합니다.
        # 여기서는 Chosen/Rejected 텍스트만 사용하고, 트레이너가 나머지를 처리하도록 합니다.
        
        # *Note*: TRL의 DPO/ARM 트레이너는 보통 prompt와 chosen/rejected 텍스트를 받아서 내부적으로 전체 시퀀스를 토크나이즈합니다.
        
        return {
            "prompt": formatted_prompt,
            "chosen": sample["response_0"],
            "rejected": sample["response_1"],
            "labels": {obj: labels[obj] for obj in obj_key}
        }
        
    return_prompt_and_responses_with_version = lambda x: return_prompt_and_responses(x, obj_key, tokenizer)

    # need to set batched=False because return_prompt_and_responses_with_version can only be applied to a single sample
    return train_dataset.map(
        return_prompt_and_responses_with_version,
        batched=False, 
        num_proc=num_proc,
        remove_columns=original_columns,
    ), test_dataset.map(
        return_prompt_and_responses_with_version,
        batched=False,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ##### chat template ends #####

if __name__ == "__main__":

    parser = TrlParser((ScriptArguments, ARMConfig))
    script_args, training_args = parser.parse_args_and_config()

    print(f'\nPreference dataset: {script_args.preference_dataset}\n')

    training_args.gradient_checkpointing_kwargs={"use_reentrant": False} # this is necessary due to https://github.com/huggingface/trl/issues/480

    # assert script_args.algorithm in ["dpo", "arm"], "algorithm must be either dpo or arm"
    if training_args.gamma > 0:
        assert training_args.loss_type == "sigmoid", "gamma (the target_reward_margin) is only supported for sigmoid loss."

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Load the preference dataset
    if script_args.preference_dataset in ["hh_rlhf"]:
        training_args.obj_key, training_args.beta_obj = [], []
        if script_args.safe_obj:
            training_args.obj_key.append('safe')
            training_args.beta_obj.append(script_args.beta_safe)
        if script_args.help_obj:
            training_args.obj_key.append('help')
            training_args.beta_obj.append(script_args.beta_help)
        if script_args.humor_obj:
            training_args.obj_key.append('humor')
            training_args.beta_obj.append(script_args.beta_humor)
        train_dataset, eval_dataset = get_hh_rlhf(dataset_name=script_args.preference_dataset, 
                                                  sanity_check=script_args.sanity_check, 
                                                  obj_key=training_args.obj_key,
                                                  tokenizer=tokenizer)
    else:
        raise ValueError(f"Invalid preference dataset: {script_args.preference_dataset}")

    print(f'\nBefore filtering. Train data size: {train_dataset.num_rows}, Test data size: {eval_dataset.num_rows}\n')

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    print(f'After filtering. Train data size: {train_dataset.num_rows}, Test data size: {eval_dataset.num_rows}\n')

    # 2. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    print(f'\n model_name_or_path: {script_args.model_name_or_path}\n')
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 3. initialize training arguments (done in DPOconfig) and peft config
    peft_config = PBLoraConfig(
        r1=script_args.lora_r,
        r2=script_args.lora_r2,
        obj_num=len(training_args.obj_key),
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. initialize the DPO/ARM trainer
    print('\n**********************************')
    training_args.pref_sample_p = script_args.pref_sample_p
    trainer = PrefARMTrainer(
        model=model,
        ref_model=None, # if not using peft, pass the model to bypass DPO Trainer check for ref model but is NOT actually used; if using peft (which is the case here), just pass None
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 5. train
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # 6. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
