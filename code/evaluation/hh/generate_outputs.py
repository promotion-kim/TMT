'''
The evaluation data is from the supplemetary material of the safe-RLHF paper: https://openreview.net/forum?id=TyFrPOKYXw
The template is for the Alpaca model in the safe-rlhf paper. Here we assume all the models (base and reward model) use the same template.
'''

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os
from collections import OrderedDict
import torch
import shutil

# from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_arithmetic import ModelArithmetic, PromptedLLM
from peft.utils.save_and_load import get_peft_model_state_dict

'''PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT'''

SYSTEM_PROMPT: str = "You are a helpful, harmless, and humorous assistant."

def _parse_dialogue(dialogue_string: str) -> list[dict]:

    messages = []

    dialogue_string = dialogue_string.strip() 

    if dialogue_string.endswith("\n\nAssistant:"):
        dialogue_string = dialogue_string[:-len("\n\nAssistant:")].strip()

    turns = dialogue_string.split("\n\nHuman:") 

    turns = [t.strip() for t in turns if t.strip()]

    for full_turn in turns:
        if "\n\nAssistant:" in full_turn:
            try:
                user_content, assistant_content = full_turn.split("\n\nAssistant:", 1)
                messages.append({"role": "user", "content": user_content.strip()})
                messages.append({"role": "assistant", "content": assistant_content.strip()})
            except ValueError:
                print(f"Warning: Could not split turn cleanly: {full_turn[:50]}...")
                continue
        else:
            messages.append({"role": "user", "content": full_turn.strip()})
    return messages

def _parse_and_template(tokenizer, raw_prompt: str, system_prompt: str) -> str:

    messages = _parse_dialogue(raw_prompt) 

    messages.insert(0, {"role": "system", "content": system_prompt})

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt

def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generation of prompts using LLMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--model_base_name_or_path',
        default="meta-llama/Llama-2-7b-chat",
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--model_parm_both_name_or_path',
        default='/path',
        type=str,
        help='the name or path of model to load from',
    )
    model_parser.add_argument(
        '--alpha_helpfulness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness + alpha_humor * M_humor',
    )
    model_parser.add_argument(
        '--alpha_harmlessness',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness + alpha_humor * M_humor',
    )
    model_parser.add_argument(
        '--alpha_humor',
        type=float,
        help='M = M_base + alpha_helpfulness * M_helpfulness + alpha_harmlessness * M_harmlessness + alpha_humor * M_humor',
    )

    model_parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='The maximum sequence length of the generation.',
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )

    model_parser.add_argument(
        '--normalize_logit',
        type=str2bool,
        default=False,
        help='If True, the temperature argument to ModelArithmetic will be enforced to 1.0, and thus the logit weights of base LLM and ARM sum up to 1; If false, when using ARM decoding, the temperature will be 1/(1+alpha_1+alpha_2).',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--datasets',
        type=str,
        default="../../data/hh/test_prompt_only.json"
    )
    
    #!
    dataset_parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to evaluate (default: all)'
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default="./results_temp",
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--cache_dir',
        type=str,
        default="./cache_temp",
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--resume',
        type=str2bool,
        default=False,
    )

    args = parser.parse_args()
    return args

################ Model Arithmetic ################
def get_model_arithmetic(model_pth_base, model_pth_reward_both, args):
    '''
    Return the decoding policy and the temperature (input to generate) that corresponds to temperature=1 w.r.t. decoding policy

    NOTE
    1. The computation of temperature is only correct when beta = beta_1 = beta_2.
    2. We do not use any system prompt for now. 
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_pth_base)
    def tinyllama_template_wrapper(system_prompt, input_string):
        # input_string은 datasets에서 온 raw_prompt (전체 대화 기록)입니다.
        return _parse_and_template(tokenizer, input_string, system_prompt)

    # note: the reward model needs to use the tokenizer of the base model for the following reasons
    # when base = llama (32000 vocab size) and arm = alpaca (might have 32001 tokens), the model_arithmetic will truncate to the tokenizer vocab size (32000) to make it work. 
    M_base = PromptedLLM(system_prompt="Not used", prompt_template=tinyllama_template_wrapper, model=model_pth_base, tokenizer=tokenizer) 
    M_reward = PromptedLLM(system_prompt="Not used", prompt_template=tinyllama_template_wrapper, model=model_pth_reward_both, tokenizer=tokenizer)

    formula = M_base + M_reward
    
    model = ModelArithmetic(formula, max_length = args.max_length)
    temperature = 0

    return model, tokenizer, temperature

################ Model Arithmetic Ends ################

if __name__ == '__main__':

    args = parse_arguments()

    model_name = f'PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor'

    ##################### output path #####################
    out_path = Path(os.path.join(args.output_dir, model_name, 'generation') + f".json")
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    print(f'Generation results will be saved in {out_path}')

    ##################### Load dataset #####################
    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)[:args.num_samples]

    ##### change pref_vec_init for reward ####
    cache_path = os.path.join(args.cache_dir, model_name)
    os.makedirs(cache_path, exist_ok=True)
    with open(f'{args.model_parm_both_name_or_path}/adapter_config.json', 'r') as f:
        config = json.load(f)

    config['pref_vec_init'] = [args.alpha_harmlessness, args.alpha_helpfulness, args.alpha_humor]

    with open(f'{cache_path}/adapter_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    shutil.copyfile(f'{args.model_parm_both_name_or_path}/adapter_model.safetensors', f'{cache_path}/adapter_model.safetensors')

    ##################### Load models #####################
    model, tokenizer, temperature = get_model_arithmetic(model_pth_base=args.model_base_name_or_path, 
                    model_pth_reward_both=cache_path,
                    args=args)
    model.eval()
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0 in the model_arithmetic generation; The logit weights of base models and potential ARMs are normalized.\n')
        temperature = 1.0
    # generate function will remove eos token
    #!
    generate = lambda prompt: model.generate_text(prompt, max_new_tokens=args.max_new_tokens, batch_size=None, temperature=temperature, top_p=1, top_k=0, do_speculation=False)[0].removesuffix(tokenizer.eos_token).strip()

    if args.normalize_logit:
        model_name += '_NormalizedLogit'

    print(f'\nModel Name: {model_name}')

    ##################### Generate responses #####################
    output_set = []
    start_index = 0

    start_time_script = time.time()
    for i in tqdm(range(start_index, len(data_evaluation))):
        prompt = data_evaluation[i]['prompt']
        # prompt_input = PROMPT_INPUT.format(input=prompt) # NOTE: the template is important
        prompt_input = prompt # the template is already used when initializing model arithmetic. 

        start = time.time()
        response = generate(prompt_input)
        elapsed = time.time() -start

        output_set.append({
            "uid": data_evaluation[i]['uid'],
            "prompt": prompt, 
            "response": response,
            "model": model_name,
            "elapsed":elapsed,
            })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)
    print(f'Generating responses finished!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/ 3600} hours for {len(output_set) - start_index} outputs')
