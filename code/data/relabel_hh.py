from datasets import Dataset, load_dataset
import os,torch, json, random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

output_dir = 'hh'
os.makedirs(f"./{output_dir}", exist_ok=True)

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

dataset = load_dataset(
    "Anthropic/hh-rlhf",
    split="train",
    num_proc=4,
)

#!
#dataset = dataset.select(range(100))

print(dataset)

new_dataset = []

model_helpful.eval()
model_harmless.eval()
model_humor.eval()

def split_prompt_response(text):
    split_token = "\n\nAssistant:"
    if split_token in text:
        parts = text.rsplit(split_token, 1)
        prompt = parts[0] + split_token
        response = parts[1].lstrip()
        return prompt, response
    else:
        print("format crashed!")
        return text, ""

with torch.no_grad():
    for num, d in tqdm(enumerate(dataset), total=len(dataset)):
        chosen = d['chosen']
        rejected = d['rejected']
        prompt, chosen_response = split_prompt_response(chosen)
        _, rejected_response = split_prompt_response(rejected)
        new_d = {'prompt': prompt, 'response_0': chosen_response, 'response_1': rejected_response}
        for idx in [0, 1]:
            input = prompt + " " +new_d[f"response_{idx}"]
            input_ids = tokenizer(input, return_tensors='pt', max_length=1024, truncation=True).to("cuda:0")
            input_ids_humor = humor_tokenizer(input, return_tensors='pt', max_length=512, truncation=True).to("cuda:0")
            new_d[f'help_score_{idx}'] = model_helpful(**input_ids).logits[0][0].item()
            new_d[f'harm_score_{idx}'] = model_harmless(**input_ids).logits[0][0].item()
            new_d[f'humor_score_{idx}'] = model_humor(**input_ids_humor).logits[0][1].item()
         
        new_d['better_response_id'] = 0 if new_d['help_score_0'] > new_d['help_score_1'] else 1
        new_d['safer_response_id'] = 0 if new_d['harm_score_0'] > new_d['harm_score_1'] else 1
        new_d['humorer_responses_id'] = 0 if new_d['humor_score_0'] > new_d['humor_score_1'] else 1

        new_dataset.append(new_d)
        
with open(f'./{output_dir}/all.json', 'w') as f:
    json.dump(new_dataset, f)

random.shuffle(new_dataset)
with open(f'./{output_dir}/train.json', 'w') as f:
    json.dump(new_dataset[:8000], f)

with open(f'./{output_dir}/dev.json', 'w') as f:
    json.dump(new_dataset[8000:8500], f)

with open(f'./{output_dir}/test.json', 'w') as f:
    json.dump(new_dataset[8500:], f)


test_prompt_only = []
for idx, d in enumerate(new_dataset[8500:]):
    test_prompt_only.append({'uid': 'eval{}'.format(idx), 'prompt': d['prompt']})

with open(f'./{output_dir}/test_prompt_only.json', 'w') as f:
    json.dump(test_prompt_only, f, ensure_ascii=False, indent=4)