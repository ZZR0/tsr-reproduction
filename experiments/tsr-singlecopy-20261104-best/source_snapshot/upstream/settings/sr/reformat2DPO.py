import json



def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        return dataset


def save_data_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

import argparse

system_prompt_chat = {
    "content": "You are a helpful, respectful, and knowledgeable assistant. Your goal is to provide accurate, concise, and relevant answers to user queries. If you are unsure about something, politely let the user know and avoid making assumptions. Always prioritize clarity and helpfulness in your responses.",
    "role": "system"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Convert data format for DPO training')
    parser.add_argument('--input_path', type=str, default="../sr/iter/0.json")
    parser.add_argument('--output_path', type=str, default="../settings/sr/data_dpo/iter0/train.json")
    args = parser.parse_args()
    return args
args = parse_args()

dataset = read_json_file(args.input_path)
dpo_data_list = []
for item in dataset:
    prompt = item['prompt']
    chosen = item['chosen_response']
    rejected = item['rejected_response']
    dpo_data = {
        "prompt": prompt, 
        "chosen": [system_prompt_chat, {"content": prompt, "role": "user"}, {"content": chosen, "role": "assistant"}], 
        "rejected": [system_prompt_chat, {"content": prompt, "role": "user"}, {"content": rejected, "role": "assistant"}]
    }
    dpo_data_list.append(dpo_data)

save_data_to_json(dpo_data_list,args.output_path)


# print("===",dataset[0])