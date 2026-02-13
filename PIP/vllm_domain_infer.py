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

Image.MAX_IMAGE_PIXELS = None

config_dicts = {
    'Qwen3-VL-8B-Instruct':       'Qwen3-VL-8B-Instruct',
    'Qwen3-VL-8B-Thinking':       'Qwen3-VL-8B-Thinking',
    'qwen3vl_235b_a22b_instruct': 'Qwen3-VL-235B-A22B-Instruct',
    'qwen3vl_235b_a22b_thinking': 'Qwen3-VL-235B-A22B-Thinking',
    'qwen3vl_30b_a3b_instruct':   'Qwen3-VL-30B-A3B-Instruct',
    'qwen3vl_30b_a3b_thinking':   'Qwen3-VL-30B-A3B-Thinking',
}

bench_dicts = {
    "blink":       "blink.jsonl",
    "mathverse":   "mathverse.jsonl",
    "mmmu_pro":    "mmmu_pro.jsonl",
    "mmstar":      "mmstar.jsonl",
    "muirbench":   "muirbench.jsonl",
    "realworldqa": "realworldqa.jsonl",
    "remi":        "remi.jsonl"
}

FILE_ROOT = 'domain_bench_infer'

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
            return None
    return images

def process_data(data, image_key='image'):
    image_paths = data[image_key]
    problem = data['question']
    content = []
    parts = problem.split('<image>')
    for i, part in enumerate(parts):
        if part.strip():
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            content.append({"type": "image", "image": image_paths[i]})
    messages = []
    messages.append({"role": "user", "content": content})

    conversation_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = load_images(image_paths)
    if images is None:
        return None
    prompt = {
        "prompt": conversation_text,
        "multi_modal_data": {"image": images}
    }
    return prompt

def vllm_infer_save(llm, sampling_params, conv_tests, dataset, benchmark_file_save):
    generates = llm.generate(conv_tests, sampling_params)
    with open(benchmark_file_save, "w", encoding="utf-8") as outfile:
        for i, output in enumerate(generates):
            responses = [res.text.strip() for res in output.outputs]
            dataset[i]['response'] = responses
            outfile.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', type=str, default='qwen2d5vl_7b_instruct', help='model to infer')
    parser.add_argument('--benchmark', type=str, default='blink', help='benchmark to infer')
    parser.add_argument('--n', type=int, default=1, help='repeat number of infer')
    parser.add_argument('--batch', type=int, default=500, help='batch of infer')
    args = parser.parse_args()

    SYS_PROMOPT = "You are an Expert to solve visual prompts. Please answer the question based on the given image and return you thinking process step by step."
    JSONL_SAVE_FILE = os.path.join(
        FILE_ROOT,
        args.benchmark + '_' + args.model + '.jsonl',
    )
    BENCH_FILE = bench_dicts[args.benchmark]

    print(f'Infer Benchmark: {args.benchmark}')
    print(f'Infer Model {args.model}')
    print(f'JSONL_SAVE_FILE: {JSONL_SAVE_FILE}')
    
    raw_datas = []
    eval_datas_id = set()
    if os.path.exists(JSONL_SAVE_FILE):
        with open(JSONL_SAVE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                eval_datas_id.add(data['id'])

    with open(BENCH_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if not data['id'] in eval_datas_id:
                raw_datas.append(data)
    
    if len(raw_datas) == 0:
        print('No new data to infer')
        exit(0)

    model_path = config_dicts[args.model]
    if '7b' or 'glm4d5v' in args.model:
        tensor_parallel_size = 4
        pipeline_parallel_size = 2
    else:
        tensor_parallel_size = 8
        pipeline_parallel_size = 1

    processor = AutoProcessor.from_pretrained(model_path)
    
    conv_tests = []
    for data in raw_datas:
        prompt = process_data(data, 'image')
        if prompt is not None:
            conv_tests.append(prompt)
    
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        n=1
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.8,
        max_model_len=97280,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )

    for i in tqdm(range(0, len(conv_tests), args.batch), desc='Infer'):
        sub_conv_tests = conv_tests[i:i + args.batch]
        sub_raw_data_list_batch = raw_datas[i:i + args.batch]

        generates = llm.generate(sub_conv_tests, sampling_params)
        
        for i, output in enumerate(generates):
            sub_raw_data_list_batch[i]['response'] = [res.text.strip() for res in output.outputs]
            sub_raw_data_list_batch[i]['conversations'] = sub_conv_tests[i]['prompt']

        with open(JSONL_SAVE_FILE, 'a', encoding='utf-8') as f:
            for entry in sub_raw_data_list_batch:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')