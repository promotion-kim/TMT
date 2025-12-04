from datasets import Dataset, load_dataset
import torch, json, random, argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generation of prompts using LLMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--path',
        default=None,
        type=str,
        help='',
    )
    return parser.parse_args()

model_path_helpful = 'Ray2333/gpt2-large-helpful-reward_model'
model_path_harmless = 'Ray2333/gpt2-large-harmless-reward_model'
model_path_humor = 'mohameddhiab/humor-no-humor'

model_helpful = AutoModelForSequenceClassification.from_pretrained(model_path_helpful, 
                                                                   torch_dtype=torch.bfloat16, 
                                                                   device_map='auto',
                                                                   num_labels=1)
model_harmless = AutoModelForSequenceClassification.from_pretrained(model_path_harmless, 
                                                                    torch_dtype=torch.bfloat16, 
                                                                    device_map='auto',
                                                                    num_labels=1)
model_humor = AutoModelForSequenceClassification.from_pretrained(model_path_humor, 
                                                                 torch_dtype=torch.bfloat16).to('cuda') #num_labels=2, no humor/humor
tokenizer = AutoTokenizer.from_pretrained(model_path_helpful)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

humor_tokenizer = AutoTokenizer.from_pretrained(model_path_humor)
if humor_tokenizer.pad_token is None:
    humor_tokenizer.pad_token = humor_tokenizer.eos_token

#SYSTEM_PROMPT: str = "You are a helpful, harmless, and humorous assistant."

args = parse_arguments()

with open(f'{args.path}/generation.json', 'r') as f:
    generation_result = json.load(f)

model_helpful.eval()
model_harmless.eval()
model_humor.eval()

reward_result = []
total_help_score, total_harm_score, total_humor_score = 0, 0, 0

template = '{input} {response}'

with torch.no_grad():
    for i in tqdm(range(len(generation_result))):
        d = generation_result[i]

        input_ids_hh = tokenizer(template.format(input=d['prompt'], response=d['response']), 
                                 return_tensors='pt',
                                 max_length=1024, 
                                 truncation=True).to("cuda:0")
        input_ids_humor = humor_tokenizer(template.format(input=d['prompt'], response=d['response']), 
                                          return_tensors='pt',
                                          max_length=512,
                                          truncation=True).to("cuda:0")
        help_score = model_helpful(**input_ids_hh).logits[0][0].item()
        harm_score = model_harmless(**input_ids_hh).logits[0][0].item()
        humor_score = model_humor(**input_ids_humor).logits[0][1].item()

        d.update({'help_score (high better)': help_score, 'harm_score (high better)': harm_score, 'humor_score (high better)': humor_score})
        reward_result.append(d)

        total_help_score += help_score
        total_harm_score += harm_score
        total_humor_score += humor_score

with open(f'{args.path}/reward_result.json', 'w') as f:
    json.dump(reward_result, f, ensure_ascii=False, indent=4)

mean_result = {'help': total_help_score/(i+1), 'harm': total_harm_score/(i+1), 'humor': total_humor_score/(i+1)}
with open(f'{args.path}/mean_result.json', 'w') as f:
    json.dump(mean_result, f, ensure_ascii=False, indent=4)