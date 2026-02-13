import json
import os
os.environ["VLLM_USE_V1"] = "1"
if "VLLM_ATTENTION_BACKEND" in os.environ:
    del os.environ["VLLM_ATTENTION_BACKEND"]
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import torch
import argparse
import re
from PIL import Image
from qwen_vl_utils import process_vision_info
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial


Image.MAX_IMAGE_PIXELS = None

config_dicts = {
    'Qwen3-VL-8B-Instruct':       'Qwen3-VL-8B-Instruct',
    'Qwen3-VL-8B-Thinking':       'Qwen3-VL-8B-Thinking',
    'qwen3vl_235b_a22b_instruct': 'Qwen3-VL-235B-A22B-Instruct',
    'qwen3vl_235b_a22b_thinking': 'Qwen3-VL-235B-A22B-Thinking',
    'qwen3vl_30b_a3b_instruct':   'Qwen3-VL-30B-A3B-Instruct',
    'qwen3vl_30b_a3b_thinking':   'Qwen3-VL-30B-A3B-Thinking',
}

FILE_ROOT = 'onlyone_bench_infer'

def read_jsonl(file_path):
    datas = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line:  
                try:
                    datas.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Unable to resolve line: {line}, Error: {e}")
    return datas

def load_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]  
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[WARNING] Unable to load image {path}: {e}")
    return images

def process_data(data, if_hard=False, image_key='image'):
    instructions = []
    if not if_hard:
        image_paths = data[image_key]
        instruct = '<image>' * len(image_paths) + '\n' + data.get("instruction", "")
        instructions.append(instruct)
    else:
        image_paths = data[image_key]
        for _ in image_paths:
            instruct = '<image>' * len(image_paths) + '\n' + data.get("instruction", "")
            instructions.append(instruct)

    prompts = []
    pattern = r'^(<image>|\n)*$'
    for problem in instructions:
        content = []
        if re.fullmatch(pattern, problem):
            for ipath in image_paths:
                content.append({"type": "image", "image": ipath})
            content.append({"type": "text", "text": ""})
        else:
            parts = problem.split('<image>')
            for i, part in enumerate(parts):
                if part.strip():
                    content.append({"type": "text", "text": part})
                if i < len(parts) - 1:
                    content.append({"type": "image", "image": image_paths[i]})
        
        messages = []
        messages.append({"role": "user", "content": content})

        conversation_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = {
            "prompt": conversation_text,
            "multi_modal_data": {"image": load_images(image_paths)}
        }
        prompts.append(prompt)
    return prompts

def vllm_infer_save(llm, sampling_params, conv_tests, dataset, benchmark_file_save):
    generates = llm.generate(conv_tests, sampling_params)
    with open(benchmark_file_save, "w", encoding="utf-8") as outfile:
        for i, output in enumerate(generates):
            responses = [res.text.strip() for res in output.outputs]
            dataset[i]['response'] = responses
            outfile.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', type=str, default='', help='model to infer')
    parser.add_argument('--n', type=int, default=1, help='repeat number of infer')
    parser.add_argument('--batch', type=int, default=32, help='batch of infer')
    args = parser.parse_args()

    JSONL_FILE = "onlyone_bench/only_one_bench_sorted_top1k5_balanced.jsonl"
    print(f'Infer Benchmark: {JSONL_FILE}')
    print(f'Infer Model {args.model}')
    text_asss = [True]
    SYS_PROMOPT = "You are an Expert to solve visual prompts. Please answer the question based on the given image and return you thinking process step by step."
    
    model_path = config_dicts[args.model]
    if '7b' or 'glm4d5v' in args.model:
        tensor_parallel_size = 4
        pipeline_parallel_size = 2
    else:
        tensor_parallel_size = 8
        pipeline_parallel_size = 1

    llm = None
    processor = None
    sampling_params = None

    for text_ass in text_asss:
        JSONL_SAVE_FILE_TEMP = os.path.join(
            FILE_ROOT,
            'temp',
            os.path.basename(JSONL_FILE).replace(".jsonl", f"_{args.model}_temp.jsonl")
        )
        if not os.path.exists(JSONL_SAVE_FILE_TEMP):
            JSONL_SAVE_FILE_TEMP = os.path.join(
                FILE_ROOT,
                os.path.basename(JSONL_FILE).replace(".jsonl", f"_{args.model}_temp.jsonl")
            )
        raw_datas = []
        eval_datas_id = set()
        print(f'JSONL_SAVE_FILE: {JSONL_SAVE_FILE_TEMP}')

        
        if os.path.exists(JSONL_SAVE_FILE_TEMP):
            with open(JSONL_SAVE_FILE_TEMP, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    eval_datas_id.add(data['id'])
                    
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if not data['id'] in eval_datas_id:
                    raw_datas.append(data)
        
        if len(raw_datas) > 0 and llm is None:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.8,
                max_model_len=60736,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
            )
            processor = AutoProcessor.from_pretrained(model_path)
            sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=1.0,
                n=1
            )

        conv_tests = []
        for data in raw_datas:
            conv_tests.extend(process_data(data, False, 'image'))
        
        for i in tqdm(range(0, len(conv_tests), args.batch), desc='Infer'):
            sub_conv_tests = conv_tests[i:i + args.batch]
            sub_raw_data_list_batch = raw_datas[i:i + args.batch]

            generates = llm.generate(sub_conv_tests, sampling_params)
            
            for i, output in enumerate(generates):
                sub_raw_data_list_batch[i]['one_response'] = [res.text.strip() for res in output.outputs]
                sub_raw_data_list_batch[i]['one_conversations'] = sub_conv_tests[i]['prompt']

            with open(JSONL_SAVE_FILE_TEMP, 'a', encoding='utf-8') as f:
                for entry in sub_raw_data_list_batch:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        if '/temp/' not in JSONL_SAVE_FILE_TEMP:
            shutil.move(JSONL_SAVE_FILE_TEMP, os.path.join(FILE_ROOT, 'temp'))
            JSONL_SAVE_FILE_TEMP = os.path.join(FILE_ROOT, 'temp', os.path.basename(JSONL_SAVE_FILE_TEMP))
        raw_datas = []
        eval_datas_id = set()
        JSONL_SAVE_FILE = os.path.join(
            FILE_ROOT,
            os.path.basename(JSONL_FILE).replace(".jsonl", f"_{args.model}.jsonl")
        )

        print(f'JSONL_SAVE_FILE: {JSONL_SAVE_FILE}')

        if os.path.exists(JSONL_SAVE_FILE):
            with open(JSONL_SAVE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    eval_datas_id.add(data['id'])

        with open(JSONL_SAVE_FILE_TEMP, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['id'] in eval_datas_id:
                    continue
                assert 'one_response' in data, f"one_response not in data {data['id']}"
                if not 'hard_responses' in data:
                    raw_datas.append(data)
        
        if len(raw_datas) > 0 and llm is None:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.8,
                max_model_len=60736,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
            )
            processor = AutoProcessor.from_pretrained(model_path)
            sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=1.0,
                n=1
            )
        elif len(raw_datas) == 0:
            continue

        VARIANT_NUM = len(raw_datas[0]['hard_image'])  
        conv_texts = []

        process_func = partial(process_data, if_hard=True, image_key='hard_image') 
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_func, raw_datas)
            for result in tqdm(results, total=len(raw_datas), desc="Processing", unit="sample"):
                conv_texts.extend(result)


        batch = args.batch * VARIANT_NUM  
        for i in tqdm(range(0, len(raw_datas), args.batch), desc='Infer'):
            sub_raw_data_list_batch = raw_datas[i:i + args.batch]
            conv_start = i * VARIANT_NUM
            conv_end = (i + args.batch) * VARIANT_NUM
            sub_conv_texts_batch = conv_texts[conv_start:conv_end]

            generates = llm.generate(sub_conv_texts_batch, sampling_params)
            raw_responses = []
            for generate in generates:
                raw_responses.extend([res.text.strip() for res in generate.outputs])
            hard_responses = [raw_responses[i:i + VARIANT_NUM] for i in range(0, len(raw_responses), VARIANT_NUM)]
            hard_conv_texts = [sub_conv_texts_batch[i:i + VARIANT_NUM] for i in range(0, len(sub_conv_texts_batch), VARIANT_NUM)]

            for i in range(0, len(hard_conv_texts)):
                hard_conv_texts[i] = [hard_conv_texts[i][j]['prompt'] for j in range(0, len(hard_conv_texts[i]))]

            for i, hard_responses_item in enumerate(hard_responses):
                sub_raw_data_list_batch[i]['hard_responses'] = hard_responses_item
                sub_raw_data_list_batch[i]['hard_conversations_list'] = hard_conv_texts[i]

            with open(JSONL_SAVE_FILE, 'a', encoding='utf-8') as f:
                for entry in sub_raw_data_list_batch:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')