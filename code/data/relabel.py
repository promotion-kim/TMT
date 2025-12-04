from datasets import Dataset, load_dataset
import torch, json, random
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore
from tqdm import tqdm

model_path_helpful = 'PKU-Alignment/beaver-7b-v1.0-reward'
model_path_harmless = 'PKU-Alignment/beaver-7b-v1.0-cost'

model_helpful = AutoModelForScore.from_pretrained(model_path_helpful, torch_dtype=torch.bfloat16, device_map='auto')
model_harmless = AutoModelForScore.from_pretrained(model_path_harmless, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path_helpful)

template = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}'

dataset = load_dataset(
    "PKU-Alignment/PKU-SafeRLHF-10K",
    split="train",
    num_proc=2,
)

print(dataset)

new_dataset = []

model_helpful.eval()
model_harmless.eval()

with torch.no_grad():
    for num, d in tqdm(enumerate(dataset)):
        new_d = {'prompt': d['prompt'], 'response_0': d['response_0'], 'response_1': d['response_1']}
        for idx in [0, 1]:
            input_ids = tokenizer(template.format(input=d['prompt'], response=d[f'response_{idx}']), 
                return_tensors='pt').to("cuda:0")
            new_d[f'help_score_{idx}'] = model_helpful(**input_ids)['end_scores'][0][0].item()
            new_d[f'harm_score_{idx}'] = model_harmless(**input_ids)['end_scores'][0][0].item()
        
        new_d['better_response_id'] = 0 if new_d['help_score_0'] > new_d['help_score_1'] else 1
        new_d['safer_response_id'] = 0 if new_d['harm_score_0'] < new_d['harm_score_1'] else 1

        new_dataset.append(new_d)
        
with open('./all.json', 'w') as f:
    json.dump(new_dataset, f)

random.shuffle(new_dataset)
with open('./train.json', 'w') as f:
    json.dump(new_dataset[:8000], f)

with open('./dev.json', 'w') as f:
    json.dump(new_dataset[8000:8500], f)

with open('./test.json', 'w') as f:
    json.dump(new_dataset[8500:], f)


test_prompt_only = []
for idx, d in enumerate(new_dataset[8500:]):
    test_prompt_only.append({'uid': 'eval{}'.format(idx), 'prompt': d['prompt']})

with open('./test_prompt_only.json', 'w') as f:
    json.dump(test_prompt_only, f, ensure_ascii=False, indent=4)