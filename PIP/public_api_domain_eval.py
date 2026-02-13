# -*- coding: utf-8 -*-
import re
import json
import argparse
from public_model_api import *
from copy import deepcopy
import numpy as np
import os
from tqdm import tqdm

JSONL_FILES = "domain_bench_infer/{bench}_{model}.jsonl"

def load_jsonl_to_conversation_list(datas, sys_prompt):
    critic_prompt = """# Role
You are an AI judging assistant, responsible for evaluating the accuracy of a model's response to a given question.

# Task
Based on the provided "correct answer" and "model's response," determine whether the "model's response" is correct.

# Input
1. **Correct Answer**: {gt_answer}
2. **Model's Response**: {response}

# Judging Criteria
1. **Core Judgment**: Focuses on whether the "model's response" is fundamentally correct compared to the **correct answer**.
2. **Format Insensitivity**:
* **Case Insensitivity**: Ignores capitalization differences in responses (e.g., "apple" and "Apple" are treated the same).
* **Synonyms/Expressions**: If the words or expressions used in the response are very similar to the "correct answer" and convey the same core meaning in the context of the question, it is considered correct.
3. **Multiple Choice Question Handling**:
* For multiple choice questions, the "correct answer" may be the letter of the answer (e.g., "C") or the specific text of that answer. * The "model's answer" is considered correct if it clearly indicates the correct option letter or expresses an answer that is consistent with the correct option content.
* If the "correct answer" provides the content of the option, and the model's answer gives the corresponding correct option letter, it is also considered correct. The reverse is also true.
4. **Chain-of-Thought (CoT) Answer Processing**:
* The "model's answer" typically includes a detailed chain of thought steps.
* You need to accurately extract the model's final conclusion or answer from these thought steps.
* The evaluation will be based on whether the model's **final answer** meets the above criteria. The detail of its thought process or the way it is expressed is not the focus of this evaluation.
5. **Focus on the Final Conclusion**: Even if there are minor flaws in the intermediate steps of the CoT, if the final conclusion is essentially consistent with the correct answer, it should be considered correct.

# Output format
For correctly answered questions, output "<result>True</result>"; for incorrect answers, output "<result>False</result>". Your final answer should be enclosed in <result></result>.
"""
    results = []
    for idx, data in enumerate(datas):
        responses = data['response']
        if responses is None:
            print(data)
        count = -1
        for ir, res in enumerate(responses):
            if not isinstance(res, list):
                res = [res]
            res = [re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL) for r in res]
            for i, r in enumerate(res):
                meta_data = {
                    "image": [], 
                    "system": sys_prompt,
                    "conversations": [
                        {
                            "from": "human",
                            "value": critic_prompt.format(gt_answer=data['answer'], response=r.replace('<image>', '\<image\>'))
                        }
                    ]
                }
                count += 1
                results.append([idx, count, meta_data])

    return results


def extract_answer_cal_metric(outputs):
    def str_to_bool(s):
        if "true" in s.lower():
            return True
        if "false" in s.lower():
            return False
        return None
    eval_results = []
    gpt_responses = []
    for item in outputs:
        res_matchs = []
        gpt_item = []
        for res in item:
            matches = re.findall(r'<result>(.*?)</result>', res, re.DOTALL)
            if len(matches) > 0:
                res_matchs.append(str_to_bool(matches[-1].strip()))
            else:
                res_matchs.append(None)
            gpt_item.append(res)
        gpt_responses.append(gpt_item)
        eval_results.append(res_matchs)
    return eval_results, gpt_responses  


def get_jsonl_files(model_arg):
    jsonls = []
    for bench in ['blink', 'mathverse', 'mmmu_pro', 'mmstar', 'muirbench', 'realworldqa', 'remi']:
        jsonls.append(JSONL_FILES.format(bench=bench, model=model_arg))
    return jsonls

def load_data(jsonl_file, jsonl_save_file):
    raw_datas = []
    eval_data_ids = set()
    eval_data = []

    if os.path.exists(jsonl_save_file):
        with open(jsonl_save_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                eval_data_ids.add(data.get('id', None))
                eval_data.append(data)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            if data.get('id', None) not in eval_data_ids:
                raw_datas.append(data)
    
    return raw_datas, eval_data

def process_and_evaluate(raw_datas, model, batch_size, jsonl_save_file):
    SYS_PROMOPT = "You are an AI assistant responsible for answering user instructions."
    all_eval_results = []
    data_list = load_jsonl_to_conversation_list(raw_datas, SYS_PROMOPT)

    len_responses = len(raw_datas[0]['response'])
    if not isinstance(raw_datas[0]['response'][0], list):
        len_res = 1
    else:
        len_res = len(raw_datas[0]['response'][0])
    eval_batch_size = len_responses * len_res
    for i in tqdm(range(0, len(raw_datas), batch_size), desc='Infer'):

        data_start_idx = i * eval_batch_size
        data_end_idx = data_start_idx + batch_size * eval_batch_size
        sub_data_list_batch = data_list[data_start_idx:data_end_idx]
        sub_raw_data_list_batch = raw_datas[i:i + batch_size]
        if len(sub_data_list_batch) == 0:
            continue
        outputs = model.infer_batch(sub_data_list_batch)
        eval_results, gpt_responses = extract_answer_cal_metric(outputs)
    
        for idx, item in enumerate(eval_results):
            sub_raw_data_list_batch[idx]['eval'] = item
            sub_raw_data_list_batch[idx]['gpt'] = gpt_responses[idx]
        all_eval_results.extend(eval_results)
        with open(jsonl_save_file, 'a', encoding='utf-8') as f:
            for entry in sub_raw_data_list_batch:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return all_eval_results

def calculate_and_print_metrics(final_eval_results, jsonl_save_file):
    print(f"{jsonl_save_file.split('/')[-1]}")
    if not isinstance(final_eval_results[0][0], list):
        arr = np.array(final_eval_results)
        arr = np.where(arr == None, np.nan, arr)
        pass_at = arr.shape[1] if arr.ndim > 1 else 1
        print(f"ACC pass@{pass_at}: {arr.any(axis=1).mean()}, total: {len(final_eval_results)}")
        if arr.ndim > 1 and arr.shape[1] > 0:
            print(np.nanmean(arr, axis=0))
    else:
        J = 3  
        all_j_results = [[] for _ in range(J)]
        for i in range(len(final_eval_results)):
            for j in range(J):
                if j < len(final_eval_results[i]) and final_eval_results[i][j] is not None:
                    all_j_results[j].append(final_eval_results[i][j])
                else:
                    all_j_results[j].append([])  
        for j in range(J):
            final_eval_results = all_j_results[j]
            max_len = max(len(x) for x in final_eval_results) if final_eval_results else 0
            if max_len == 0:
                print(f"j={j}: ACC pass@0: 0.0, total: {len(final_eval_results)}")
                continue
            padded = []
            for seq in final_eval_results:
                if len(seq) == 0:
                    padded.append([None] * max_len)
                else:
                    padded.append(list(seq) + [None] * (max_len - len(seq)))
            arr = np.array(padded, dtype=object)
            arr = np.where(arr == None, np.nan, arr.astype(float))
            total_samples = len(final_eval_results)
            if arr.size == 0 or np.all(np.isnan(arr)):
                print(f"j={j}: ACC pass@{max_len}: 0.0, total: {total_samples}")
                if max_len > 0:
                    print(np.array([np.nan] * max_len))
                continue
            any_correct = np.any(arr == 1.0, axis=1)
            valid_rows = ~np.all(np.isnan(arr), axis=1)
            if np.any(valid_rows):
                pass_at_k = np.sum(any_correct & valid_rows) / np.sum(valid_rows)
            else:
                pass_at_k = 0.0
            print(f"j={j}: ACC pass@{max_len}: {pass_at_k}, total: {total_samples}")
            if arr.shape[1] > 0:
                col_acc = np.nanmean(arr, axis=0)
                print(col_acc)

def parse_arguments():
    parser = argparse.ArgumentParser(description="bench")
    parser.add_argument('--model', type=str, default='qwen2d5_vl_7b_mgdo_gspo_rdn_batch192_step300')
    parser.add_argument('--batch', type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("######################################################################")
    print(f'Eval model {args.model}')
    print("######################################################################")

    config = {
        "api_key": "xxx",
        "model_name": "xxx",
    }
    model = GPT_EVAL(**config)

    jsonl_files = get_jsonl_files(args.model)

    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            continue
        
        print(jsonl_file)
        jsonl_save_file = jsonl_file.replace('.jsonl', '_eval.jsonl')

        raw_datas, eval_data = load_data(jsonl_file, jsonl_save_file)
        final_eval_results = [data['eval'] for data in eval_data if data.get('eval', None) is not None]
        
        if len(raw_datas) > 0:
            new_eval_results = process_and_evaluate(raw_datas, model, args.batch, jsonl_save_file)
            final_eval_results.extend(new_eval_results)
        
        calculate_and_print_metrics(final_eval_results, jsonl_save_file)


if __name__ == "__main__":
    main()