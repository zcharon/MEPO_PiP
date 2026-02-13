# -*- coding: utf-8 -*-
import re
import json
import argparse
from public_model_api import *
from copy import deepcopy
import numpy as np
import os
from tqdm import tqdm
import shutil

JSONL_FILES =  [
    "onlyone_bench_infer/only_one_bench_sorted_top1k5_balanced_{model}.jsonl"
]

TEMP_FILE_ROOT = 'onlyone_bench_infer/temp'

def load_jsonl_to_conversation_list(datas, sys_prompt, if_hard=True):
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
        if not if_hard:
            responses = data['one_response']
        else:
            responses = data['hard_responses']
        if responses is None:
            print(data)
        count = -1
        for res in responses:
            if not isinstance(res, list):
                res = [res]
            for r in res:
                pattern = r'<(think|reason)>.*?</\1>'
                cleaned_r = re.sub(pattern, '', r, flags=re.DOTALL)
                meta_data = {
                    "image": [],  
                    "system": sys_prompt,
                    "conversations": [
                        {
                            "from": "human",
                            "value": critic_prompt.format(gt_answer=data['answer'], response=cleaned_r.replace('<image>', '\<image\>').replace('<video>', '\<video\>').replace('<audio>', '\<audio\>'))
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


def get_jsonl_files(model):
    return [jsonl_file.format(model=model) for jsonl_file in JSONL_FILES]


def load_data(jsonl_file, jsonl_save_file):
    raw_datas = []
    eval_data_ids = set()
    eval_data = []
    type_list = []
    domain_list = []

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
            type_list.append(data['type'])
            domain_list.append(data['domain'])
            if data.get('id', None) not in eval_data_ids:
                raw_datas.append(data)
    
    return raw_datas, eval_data, type_list, domain_list


def process_and_evaluate_one(raw_datas, model, batch_size, jsonl_save_file):
    SYS_PROMOPT = "You are an AI assistant responsible for answering user instructions."
    all_eval_results = []
    data_list = load_jsonl_to_conversation_list(raw_datas, SYS_PROMOPT, False)

    len_responses = len(raw_datas[0]['one_response'])
    if not isinstance(raw_datas[0]['one_response'][0], list):
        len_res = 1
    else:
        len_res = len(raw_datas[0]['one_response'])

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
            sub_raw_data_list_batch[idx]['one_eval'] = item
            sub_raw_data_list_batch[idx]['one_gpt'] = gpt_responses[idx]
        all_eval_results.extend(eval_results)
        with open(jsonl_save_file, 'a', encoding='utf-8') as f:
            for entry in sub_raw_data_list_batch:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return all_eval_results


def process_and_evaluate_hard(raw_datas, model, batch_size, jsonl_save_file):
    SYS_PROMOPT = "You are an AI assistant responsible for answering user instructions."
    all_eval_results = []
    data_list = load_jsonl_to_conversation_list(raw_datas, SYS_PROMOPT, True)

    if not isinstance(raw_datas[0]['hard_responses'][0], list):
        # raw_datas = [[item] for item in raw_datas]
        hard_responses = [[item] for item in raw_datas[0]['hard_responses']]
        len_res = len(hard_responses)
    else:
        len_res = len(raw_datas[0]['hard_responses'])

    eval_batch_size = len_res
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
            n = len(item) // 3
            item = [item[i * n : (i + 1) * n] for i in range(3)]
            gpt_responses[idx] = [gpt_responses[idx][i * n : (i + 1) * n] for i in range(3)]
            sub_raw_data_list_batch[idx]['hard_eval'] = item
            sub_raw_data_list_batch[idx]['hard_gpt'] = gpt_responses[idx]
        all_eval_results.extend(eval_results)
        with open(jsonl_save_file, 'a', encoding='utf-8') as f:
            for entry in sub_raw_data_list_batch:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return all_eval_results


def calculate_and_print_metrics(final_eval_one_results, final_eval_hard_results, type_list, domain_list):
    N = len(final_eval_one_results)
    assert N == len(final_eval_hard_results) == len(type_list) == len(domain_list), "All lists must have the same length."
    assert all(len(hard) == 3 for hard in final_eval_hard_results), "Each hard result must have 3 perturbations."
    assert all(len(hard[j]) == 1 for hard in final_eval_hard_results for j in range(3)), "Each perturbation result must be a single-element list."

    # Extract flat correctness
    correct_one = [res[0] for res in final_eval_one_results]

    # Helper: get perturbation correctness as flat list per sample
    hard_correct_flat = []
    for i in range(N):
        hard_correct_flat.append([final_eval_hard_results[i][j][0] for j in range(3)])

    # --- Overall Metrics ---
    for i in range(len(correct_one)):
        if correct_one[i] is None:
            print(i, 'eval result is None')
            correct_one[i] = False  

    Acc = sum(correct_one) / N

    rr_indices = [i for i, t in enumerate(type_list) if t == "RR"]
    cr_indices = [i for i, t in enumerate(type_list) if t == "CR"]
    N_r, N_c = len(rr_indices), len(cr_indices)
    RRA = sum(correct_one[i] for i in rr_indices) / N_r if N_r > 0 else float('nan')
    CRA = sum(correct_one[i] for i in cr_indices) / N_c if N_c > 0 else float('nan')

    # SR (overall)
    orig_correct_indices = [i for i in range(N) if correct_one[i]]
    N_correct = len(orig_correct_indices)
    if N_correct > 0:
        SR = sum(1 for i in orig_correct_indices if all(hard_correct_flat[i])) / N_correct
    else:
        SR = float('nan')

    # --- Domain-wise Metrics ---
    from collections import defaultdict
    domain_indices = defaultdict(list)
    for i, domain in enumerate(domain_list):
        domain_indices[domain].append(i)

    domain_metrics = {}
    for domain, indices in domain_indices.items():
        dN = len(indices)
        d_correct = [correct_one[i] for i in indices]
        dAcc = sum(d_correct) / dN

        # RRA in domain
        d_rr_idx = [i for i in indices if type_list[i] == "RR"]
        dN_r = len(d_rr_idx)
        dRRA = sum(correct_one[i] for i in d_rr_idx) / dN_r if dN_r > 0 else float('nan')

        # CRA in domain
        d_cr_idx = [i for i in indices if type_list[i] == "CR"]
        dN_c = len(d_cr_idx)
        dCRA = sum(correct_one[i] for i in d_cr_idx) / dN_c if dN_c > 0 else float('nan')

        # SR in domain: conditional on original correct within domain
        d_orig_correct = [i for i in indices if correct_one[i]]
        dN_corr = len(d_orig_correct)
        if dN_corr > 0:
            dSR = sum(1 for i in d_orig_correct if all(hard_correct_flat[i])) / dN_corr
        else:
            dSR = float('nan')

        domain_metrics[domain] = {
            "Acc": dAcc,
            "RRA": dRRA,
            "CRA": dCRA,
            "SR": dSR,
            "N": dN,
            "N_r": dN_r,
            "N_c": dN_c,
            "N_correct": dN_corr
        }

    # --- Print Results ---
    print(f"Overall Prediction Accuracy (Acc): {Acc:.4f}")
    print(f"Overall Region-Referenced Accuracy (RRA): {RRA:.4f} (N_r={N_r})" if N_r > 0 else "Overall RRA: N/A")
    print(f"Overall Composite Reasoning Accuracy (CRA): {CRA:.4f} (N_c={N_c})" if N_c > 0 else "Overall CRA: N/A")
    print(f"Overall Structural Robustness (SR): {SR:.4f} (N_correct={N_correct})" if N_correct > 0 else "Overall SR: N/A")

    print("\n" + "="*60)
    print("Domain-wise Metrics:")
    print("="*60)

    for domain in sorted(domain_metrics.keys()):
        m = domain_metrics[domain]
        print(f"\nDomain: {domain} (N={m['N']})")
        print(f"  ACC : {m['Acc']:.4f}")
        if m['N_r'] > 0:
            print(f"  RRA : {m['RRA']:.4f} (N_r={m['N_r']})")
        else:
            print(f"  RRA : N/A")
        if m['N_c'] > 0:
            print(f"  CRA : {m['CRA']:.4f} (N_c={m['N_c']})")
        else:
            print(f"  CRA : N/A")
        if m['N_correct'] > 0:
            print(f"  SR  : {m['SR']:.4f} (N_correct={m['N_correct']})")
        else:
            print(f"  SR  : N/A")

    return {
        "overall": {"Acc": Acc, "RRA": RRA, "CRA": CRA, "SR": SR},
        "domain": domain_metrics
    }
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="评测 bench 脚本")
    parser.add_argument('--model', type=str, default='mimo_vl_7b_sft_2508_mgdo_rdn_batch192_gamma_10_step200')
    parser.add_argument('--batch', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_arguments()
    print("######################################################################")
    print(f'Eval Model {args.model}')
    print("######################################################################")

    config = {
        "api_key": "**",
        "model_name": "**",
    }
    model = GPT_EVAL(**config)

    jsonl_files = get_jsonl_files(args.model)

    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            continue
        
        print(jsonl_file)
        jsonl_save_file = jsonl_file.replace('.jsonl', '_eval.jsonl')
        file_name = os.path.basename(jsonl_file)
        jsonl_save_file_temp = os.path.join(TEMP_FILE_ROOT, file_name.replace('.jsonl', '_eval_temp.jsonl'))
        if not os.path.exists(jsonl_save_file_temp):
            jsonl_save_file_temp = jsonl_save_file.replace('.jsonl', '_temp.jsonl')
        
        #############################################################
        #                OnlyOne_Bench_One_Eval                     #
        #############################################################
        print(f"OnlyOne_Bench_One_Eval")
        raw_datas, eval_data, _, _ = load_data(jsonl_file, jsonl_save_file_temp)
        final_eval_one_results = [data['one_eval'] for data in eval_data if data.get('one_eval', None) is not None]
        
        if len(raw_datas) > 0:
            new_eval_one_results = process_and_evaluate_one(raw_datas, model, args.batch, jsonl_save_file_temp)
            final_eval_one_results.extend(new_eval_one_results)

        if not os.path.exists(os.path.join(TEMP_FILE_ROOT, file_name.replace('.jsonl', '_eval_temp.jsonl'))):
            shutil.move(jsonl_save_file_temp, TEMP_FILE_ROOT)
            jsonl_save_file_temp = os.path.join(TEMP_FILE_ROOT, file_name.replace('.jsonl', '_eval_temp.jsonl'))

        #############################################################
        #                OnlyOne_Bench_Hard_Eval                    #
        #############################################################
        print(f"OnlyOne_Bench_Hard_Eval")
        raw_datas, eval_data, type_list, domain_list = load_data(jsonl_save_file_temp, jsonl_save_file)
        final_eval_hard_results = [data['hard_eval'] for data in eval_data if data.get('hard_eval', None) is not None]
        
        if len(raw_datas) > 0:
            new_eval_hard_results = process_and_evaluate_hard(raw_datas, model, args.batch, jsonl_save_file)
            final_eval_hard_results.extend(new_eval_hard_results)

        calculate_and_print_metrics(final_eval_one_results, final_eval_hard_results, type_list, domain_list)


if __name__ == "__main__":
    main()