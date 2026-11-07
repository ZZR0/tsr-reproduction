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
    parser.add_argument('--input_path_chosen', type=str, default="../tsr/iter/sepaJudge/0_chosen_judgement.json")
    parser.add_argument('--input_path_rejected', type=str, default="../tsr/iter/sepaJudge/0_rejected_judgement.json")
    parser.add_argument('--output_path', type=str, default="../settings/tsr/data_dpo/iter0_tmp/train.json")
    args = parser.parse_args()
    return args
args = parse_args()

dataset = read_json_file(args.input_path_chosen)
dataset2 = read_json_file(args.input_path_rejected)

dpo_data_list = []
leng = len(dataset)
leng2 = len(dataset2)
if leng != leng2:
    print("DATASET_CHOSEN != DATASET_REJECTED LEN")
    sys.exit(1)

for i in range(leng):
    item1 = dataset[i]
    item2 = dataset2[i]
    prompt = item1['prompt']
    prompt2 = item2['prompt']
    if prompt != prompt2:
        print("DATASET_CHOSEN prompt != DATASET_REJECTED prompt2")
        sys.exit(1) 

    index1_max = item1['scores'].index(max(item1['scores']))
    score1_max = item1['scores'][index1_max] 
    chosen = item1['responses'][index1_max]  


    index1 = item1['scores'].index(min(item1['scores']))
    score1 = item1['scores'][index1]    

    index2 = item2['scores'].index(min(item2['scores']))
    score2 = item2['scores'][index2]    

    if score1 <= score2:
        rejected = item1['responses'][index1]
        rejected_score = score1
    else:
        rejected = item2['responses'][index2]
        rejected_score = score2


    if chosen == None or rejected == None:
        continue

    dpo_data = {
        "prompt": prompt, 
        "chosen": [system_prompt_chat, {"content": prompt, "role": "user"}, {"content": chosen, "role": "assistant"}], 
        "rejected": [system_prompt_chat, {"content": prompt, "role": "user"}, {"content": rejected, "role": "assistant"}]
    }
    dpo_data_list.append(dpo_data)

save_data_to_json(dpo_data_list,args.output_path)