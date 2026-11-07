from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from scipy.optimize import minimize
import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
import re
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_jsonl_file(file_path):
    dataset = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset


class SelfRewardGenerator:
    def __init__(self, model_path: str, world_size: int):
        # Load the base model
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=world_size,
        )
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop='<|end_of_text|>'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    
    def format_chat(self, chat, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt
        )


    def format_chat_handly(self, chat, eos_token="<|end_of_text|>", add_generation_prompt=True):
        system_messages = []
        user_messages = []
        assistant_messages = []

        # Separate messages by role
        for message in chat:
            role = message['role']
            content = message['content']

            if role == 'system':
                system_messages.append(f"<|system|>\n{content}{eos_token}\n")
            elif role == 'user':
                user_messages.append(f"<|user|>\n{content}{eos_token}\n")
            elif role == 'assistant':
                assistant_messages.append(f"<|assistant|>\n{content}{eos_token}\n")

        # Concatenate messages in the desired order, skipping empty lists
        formatted_chat = (
            "".join(system_messages) +
            "".join(user_messages) +
            "".join(assistant_messages)
        )

        # Add generation prompt if enabled
        if add_generation_prompt:
            formatted_chat += "<|assistant|>\n"

        return formatted_chat
        
    def construct_chat_eval(self, prompt: str, type: str) -> list:
        """Construct a chat conversation with system and user messages"""
        _chat = [
            {"role": "system", "content": "You are a helpful, respectful, and knowledgeable assistant. Your goal is to provide accurate, concise, and relevant answers to user queries. If you are unsure about something, politely let the user know and avoid making assumptions. Always prioritize clarity and helpfulness in your responses."},
            {"role": "user", "content": prompt},
        ]
        _eval = [
            {"role": "system", "content": "You are a helpful and precise evaluator. Provide objective, constructive, and detailed feedback. Be concise, clear, and tailored to the context. Your goal is to support refinement and informed decision-making."},
            {"role": "user", "content": prompt},
        ]
        if type == "eval":
            return _eval
        return _chat

    def _format_prompt(self, content: str) -> str:
        """Format content with chat markers"""
        return f"<|system|>\n<|end_of_text|>\n<|user|>\n{content}\n<|end_of_text|>\n<|assistant|>\n"

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_texts = [x.outputs[0].text for x in outputs]
        return generated_texts


    def build_eval_user_prompt(self, question: str, response: str) -> str:
        prompt = f"""
Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:
- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {question}
<response>{response}</response>

After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria.
"""
        return prompt.strip()


    def generate_all_responses(self, prompts: List[str], num_responses: int = 7, type: str = "chat") -> List[List[str]]:
        """Generate multiple responses for each prompt in batch"""
        all_generate_prompts = []
        for prompt in prompts:
            chat_eval = self.construct_chat_eval(prompt, type)
            formatted_prompt = self.format_chat(chat_eval)
            # formatted_prompt = self._format_prompt(prompt)
            all_generate_prompts.extend([formatted_prompt] * num_responses)
        
        print("all_generate_prompts[0]...",all_generate_prompts[0])
        print("Generating responses...")
        responses_flat = self.generate_batch(all_generate_prompts)
        # Group responses by prompts
        responses = [responses_flat[i * num_responses:(i + 1) * num_responses] for i in range(len(prompts))]
        return responses

    def generate_all_judgments(self, prompts: List[str], responses: List[List[str]], num_judgments: int = 5, type: str = "eval") -> List[List[List[str]]]:
        """Generate multiple judgments for each response in batch"""
        all_judge_prompts = []
        for prompt, response_list in zip(prompts, responses):
            for response in response_list:
                eval_user_prompt = self.build_eval_user_prompt(prompt, response)
                chat_eval = self.construct_chat_eval(eval_user_prompt, type)
                formatted_judge_prompt = self.format_chat(chat_eval)
                all_judge_prompts.extend([formatted_judge_prompt] * num_judgments)
        
        print("all_judgement_prompts[0]...",all_judge_prompts[0])
        print("Generating judgments...")
        judgments_flat = self.generate_batch(all_judge_prompts)
        # Group judgments by responses and prompts
        judgments = []
        idx = 0
        for _ in prompts:
            judgments_per_prompt = []
            for _ in range(len(responses[0])):  # assuming all prompts have same number of responses
                judgments_per_response = judgments_flat[idx:idx + num_judgments]
                judgments_per_prompt.append(judgments_per_response)
                idx += num_judgments
            judgments.append(judgments_per_prompt)
        return judgments


    def extract_scores(self, judgments: List[str]) -> List[float]:
        """Extract numerical scores from judgments"""
        scores = []
        pattern = r"Score:\s*(\d+(?:\.\d+)?)"
        for judgment in judgments:
            match = re.search(pattern, judgment)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 5:
                        scores.append(score)
                    return score
                except ValueError:
                    return None
        return scores


    def select_with_score(self, responses: List[str], scores: List[float]) -> Tuple[str, str]:
        """Select chosen and rejected responses with length control"""
        s_max = max(scores)
        s_min = min(scores)
        
        chosen = responses[np.argmax(scores)]
        rejected = responses[np.argmin(scores)]
        
        return chosen, rejected

    def process_all(self, prompts: List[str], responses: List[List[str]], judgments: List[List[List[str]]]) -> List[Dict]:
        """Process all prompts with their responses, judgments"""
        results = []
        for idx, prompt in enumerate(prompts):
            response_list = responses[idx]
            judgments_list = judgments[idx]
            
            # Calculate average scores for each response
            avg_scores = []
            for j_list in judgments_list:
                scores = self.extract_scores(j_list)
                avg_score = np.mean(scores) if scores else 0
                avg_scores.append(avg_score)
            
            # Select chosen and rejected responses with length control
            chosen_response, rejected_response = self.select_with_score(response_list, avg_scores)
            
            
            result = {
                "prompt": prompt,
                "responses": response_list,
                "judgments": judgments_list,
                "scores": avg_scores,
                "chosen_response": chosen_response,
                "rejected_response": rejected_response,
            }
            results.append(result)
        return results

    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama-3-8B-sft/')
    parser.add_argument('--output_dir', type=str, default='iter/iterJudge')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--input_dir', type=str, default='/iter1.json')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', type=str2bool, default=False)
    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Initialize meta rewarding system
    self_reward = SelfRewardGenerator(args.model, args.world_size)

    # Load data
    print('---------Load Data---------')
    data = read_jsonl_file(args.input_dir)


    prompts = [item["prompt"] for item in data]
    if args.test:
        prompts = prompts[:50]
    # Generate all responses
    print('---------Generate all responses---------')
    responses = self_reward.generate_all_responses(prompts, num_responses=7, type="chat")

    # Generate all judgments
    judgments = self_reward.generate_all_judgments(prompts, responses, num_judgments=1, type="eval")


    # Process all data to compute final results
    results = self_reward.process_all(prompts, responses, judgments)

    # Save results
    output_file = os.path.join(args.output_dir, f"{args.iter}.json")
    self_reward.save_results(results, output_file)


if __name__ == "__main__":
    main()
