import io
import os
import re
import json
import requests
from volcenginesdkarkruntime import Ark
import time
import math
import openai
import base64
import functools
from copy import deepcopy
import torch
from PIL import Image
import random
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm

openai_client = openai.AzureOpenAI(
    azure_endpoint="xxx",
    api_version="xxx",
    api_key="xxx",
) 


def encode_image(image_path, min_pixels, max_pixels, resample=Image.LANCZOS):
    Image.MAX_IMAGE_PIXELS = 300000000
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    width, height = image.size
    current_pixels = width * height
    if current_pixels < min_pixels:
        scale_factor = math.sqrt(min_pixels / current_pixels)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), resample)
    elif current_pixels > max_pixels:
        scale_factor = math.sqrt(max_pixels / current_pixels)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), resample)
    jpeg_buffer = io.BytesIO()
    image.convert('RGB').save(jpeg_buffer, format='JPEG')
    jpeg_bytes = jpeg_buffer.getvalue()
    jpeg_base64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
    return jpeg_base64_string


def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')


def convert_input_eval(input_list, min_pixels, max_pixels, is_encode_image: bool = False):
    output_data = []
    for idx, data_dict in enumerate(input_list):
        data_dict = deepcopy(data_dict)
        messages = []

        if "system" in data_dict and data_dict["system"]:
            messages.append({
                "role": "system",
                "content": data_dict["system"],
            })
        id = data_dict[0]
        ir = data_dict[1]

        image = data_dict[2].get("image", [])
        audio = data_dict[2].get("audio", [])
        video = data_dict[2].get("video", [])
        for conv in data_dict[2]["conversations"]:
            if conv["from"] == "human":
                if "<image>" in conv["value"] or \
                    "<audio>" in conv["value"] or \
                        "<video>" in conv["value"]:
                    assert len(data_dict[2].get("image", [])) == conv["value"].count("<image>"), "{}: The number of images {} is not equal to the number of <image> {} in the text.".format(
                        idx,
                        len(data_dict[2]["image"]),
                        conv["value"].count("<image>")
                    )
                    assert len(data_dict[2].get("audio", [])) == conv["value"].count("<audio>"), "{}: The number of audios {} is not equal to the number of <audio> {} in the text.".format(
                        idx,
                        len(data_dict[2]["audio"]),
                        conv["value"].count("<audio>")
                    )
                    assert len(data_dict[2].get("video", [])) == conv["value"].count("<video>"), "{}: The number of videos {} is not equal to the number of <video> {} in the text.".format(
                        idx,
                        len(data_dict[2]["video"]),
                        conv["value"].count("<video>")
                    )

                    split_list = re.split(r'(<image>|<audio>|<video>)', conv["value"])
                    split_list = [s for s in split_list if s]

                    content_list = []
                    for split_str in split_list:
                        if split_str == "<image>":
                            if is_encode_image:
                                base64_image = encode_image(
                                    image.pop(0),
                                    min_pixels,
                                    max_pixels,
                                )
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                })
                            else:
                                content_list.append({
                                    "type": "image",
                                    "image": image.pop(0),
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                })
                        elif split_str == "<audio>":
                            base64_audio = encode_audio(
                                audio.pop(0)
                            )
                            content_list.append({
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"data:audio/mpeg;base64,{base64_audio}"
                                }
                            })
                        elif split_str == "<video>":
                            base64_video = encode_video(
                                video.pop(0)
                            )
                            content_list.append({
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{base64_video}"
                                }
                            })
                        else:
                            content_list.append({
                                "type": "text",
                                "text": split_str
                            })
                else:
                    content_list = conv["value"]
                
                messages.append({
                    "role": "user",
                    "content": content_list
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
            else:
                raise ValueError("No such role: {}".format(conv["from"]))
        
        output_data.append([id, ir, {
            "messages": messages,
        }])

    return output_data


def convert_input(input_list, min_pixels, max_pixels, is_encode_image: bool = False):
    output_data = []
    for idx, data_dict in enumerate(input_list):
        data_dict = deepcopy(data_dict)
        messages = []

        if "system" in data_dict and data_dict["system"]:
            messages.append({
                "role": "system",
                "content": data_dict["system"],
            })
        # import pdb; pdb.set_trace()
        image = data_dict.get("image", [])
        audio = data_dict.get("audio", [])
        video = data_dict.get("video", [])
        for conv in data_dict["conversations"]:
            if conv["from"] == "human":
                if "<image>" in conv["value"] or \
                    "<audio>" in conv["value"] or \
                        "<video>" in conv["value"]:
                    assert len(data_dict.get("image", [])) == conv["value"].count("<image>"), "{}: The number of images {} is not equal to the number of <image> {} in the text.".format(
                        idx,
                        len(data_dict["image"]),
                        conv["value"].count("<image>")
                    )
                    assert len(data_dict.get("audio", [])) == conv["value"].count("<audio>"), "{}: The number of audios {} is not equal to the number of <audio> {} in the text.".format(
                        idx,
                        len(data_dict["audio"]),
                        conv["value"].count("<audio>")
                    )
                    assert len(data_dict.get("video", [])) == conv["value"].count("<video>"), "{}: The number of videos {} is not equal to the number of <video> {} in the text.".format(
                        idx,
                        len(data_dict["video"]),
                        conv["value"].count("<video>")
                    )

                    split_list = re.split(r'(<image>|<audio>|<video>)', conv["value"])
                    split_list = [s for s in split_list if s]

                    content_list = []
                    for split_str in split_list:
                        if split_str == "<image>":
                            if is_encode_image:
                                base64_image = encode_image(
                                    image.pop(0),
                                    min_pixels,
                                    max_pixels,
                                )
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                })
                            else:
                                content_list.append({
                                    "type": "image",
                                    "image": image.pop(0),
                                    "min_pixels": min_pixels,
                                    "max_pixels": max_pixels,
                                })
                        elif split_str == "<audio>":
                            base64_audio = encode_audio(
                                audio.pop(0)
                            )
                            content_list.append({
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"data:audio/mpeg;base64,{base64_audio}"
                                }
                            })
                        elif split_str == "<video>":
                            base64_video = encode_video(
                                video.pop(0)
                            )
                            content_list.append({
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{base64_video}"
                                }
                            })
                        else:
                            content_list.append({
                                "type": "text",
                                "text": split_str
                            })
                else:
                    content_list = conv["value"]
                
                messages.append({
                    "role": "user",
                    "content": content_list
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
            else:
                raise ValueError("No such role: {}".format(conv["from"]))
        
        output_data.append({
            "messages": messages,
        })

    return output_data


class Gemieni25Pro:
    def __init__(self, *args, **kwargs) -> None:
        self.api_key = kwargs.get(
            "api_key",
            os.environ.get(
                "OPENAI_API_KEY",
                "xxx"
            )
        )
        self.model_name = kwargs.get(
            "model_name",
            "xxx"
        )
        self.api_version = kwargs.get(
            "api_version",
            "xxx",
        )
        self.azure_endpoint = kwargs.get(
            "azure_endpoint",
            "xxx",
        )
        self.client = openai.AzureOpenAI(
            azure_endpoint="xxx",
            api_version="xxx",
            api_key="xxx",
        )  # QPM=300
    
    def request_gemeni(self, messages: tuple, max_tokens: int, temperature: float, n: int,
                   model: str = "gemini-2.5-pro-preview-03-25",
                   max_retries: int = 10, retry_delay: int = 10, base_delay=1.5,
                   request_timeout: int = 120) -> list:
        i, messages = messages
        responses = []
        for _ in range(n):
            retries = 0
            while retries < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        timeout=request_timeout,
                    )
                    responses.append(completion.choices[0].message.content)
                    break
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        responses.append("ERROR")
                        break
                    exponential_delay = 0
                    jitter = random.uniform(0, 15)
                    total_delay = exponential_delay + jitter
                    print(f"[Request Retry] Attempt {retries}/{max_retries} failed: {e}. Retry in {total_delay:.2f}s")
                    time.sleep(total_delay)
        return i, responses
    
    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict) -> list:
        if param_dict is None:
            param_dict = {
                "request_timeout": 60
            }

        request_new = functools.partial(
            self.request_gemeni,
            max_tokens=param_dict.get("max_new_tokens", 32786),
            temperature=param_dict.get("temperature", 1.0),
            n=param_dict.get("n", 1),
            model=self.model_name,
            max_retries=100,
            retry_delay=1.5,
            request_timeout=param_dict.get("request_timeout", 120), 
        )

        data_list = convert_input(
            data_list,
            param_dict.get("min_pixels", 1 * 1),
            param_dict.get("max_pixels", 4096 * 4096),
            is_encode_image=False,
        )

        input_list = []
        for data_dict in data_list:
            data_dict = deepcopy(data_dict)
            for msg_dict in data_dict["messages"]:
                content = msg_dict["content"]
                if isinstance(content, list):
                    for content_dict in content:
                        if content_dict["type"] == "image":
                            base64_image = encode_image(
                                content_dict["image"],
                                content_dict.get("min_pixels", 1 * 1),
                                content_dict.get("max_pixels", 4096 * 4096),
                            )
                            content_dict["type"] = "image_url"
                            content_dict["image_url"] = {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                            del content_dict["image"]

            input_list.append(data_dict)
        
        results = []
        with ThreadPoolExecutor(max_workers=param_dict.get("max_workers", 8)) as executor:
            future_to_index = {
                executor.submit(request_new, (i, arg["messages"])): i 
                for i, arg in enumerate(input_list)
            }
            
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing requests"):
                try:
                    result = future.result()
                    index = future_to_index[future]
                    results.append((index, result))
                except Exception as e:
                    index = future_to_index[future]
                    results.append((index, f"Error: {e}"))

        final_results = [result for _, result in sorted(results)]
        
        return final_results


class GPT410414_EVAL:
    def __init__(self, *args, **kwargs) -> None:
        self.api_key = kwargs.get(
            "api_key",
            os.environ.get(
                "OPENAI_API_KEY",
                "xxx"
            )
        )
        self.model_name = kwargs.get(
            "model_name",
            "xxx"
        )
        self.api_version = kwargs.get(
            "api_version",
            "xxx",
        )
        self.azure_endpoint = kwargs.get(
            "azure_endpoint",
            "xxx",
        )
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key
        )
    

    def request(self, messages: tuple, max_tokens: int, temperature: float, n: int, model: str, max_retries: int = 600, retry_delay: int = 60, base_delay: float = 1.5):
        id, ir, messages = messages
        messages = messages["messages"]
        responses = []
        for k in range(n):
            retries = 0
            while retries < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=1,
                        messages=messages,
                    )

                    responses.append(completion.choices[0].message.content)
                    break

                except Exception as e:
                    retries += 1
                    exponential_delay = (2 ** retries) * base_delay
                    jitter = random.uniform(0, 100)
                    total_delay = exponential_delay + jitter

                    print(
                        f"[sample {k+1}] attempt {retries} failed: {e}. "
                    )
                    time.sleep(total_delay)
                    if retries == max_retries:
                        responses.append("ERROR")
        return id, ir, responses
    

    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict = None) -> list:
        if param_dict is None:
            param_dict = {}

        request_single_thread = functools.partial(
            self.request,
            max_tokens=param_dict.get("max_new_tokens", 16384),
            temperature=param_dict.get("temperature", None),
            n=param_dict.get("n", 1),
            model=self.model_name,
            max_retries=30,
            base_delay=1.5,
        )

        data_list_convert = convert_input_eval(
            data_list,
            min_pixels=param_dict.get("min_pixels", 1 * 1),
            max_pixels=param_dict.get("max_pixels", 4096 * 4096),
            is_encode_image=True,
        )
        
        results = []
        with ThreadPoolExecutor(max_workers=param_dict.get("max_workers", 10)) as executor:
            future_to_index = {
                executor.submit(request_single_thread, data_dict): i
                for i, data_dict in enumerate(data_list_convert)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Processing requests",
                unit="req"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx, ir = future_to_index[future]
                    results.append((idx, ir, f"Error: {e}"))

        from collections import defaultdict
        grouped = defaultdict(dict)
        for idx, ir, res in results:
            grouped[idx][ir] = res

        ir_counts = {idx: len(irs) for idx, irs in grouped.items()}
        if len(set(ir_counts.values())) != 1:
            raise ValueError(f"idx != ir: {ir_counts}")

        max_idx = max(grouped.keys())
        max_ir = max(ir for irs in grouped.values() for ir in irs.keys())

        final_results = []
        for i in range(data_list[0][0], data_list[-1][0] + 1):
            if i not in grouped:
                raise ValueError(f"don't have idx={i} results")  
            inner = [grouped[i][ir] for ir in range(max_ir + 1)]
            flat = [item for sublist in inner for item in sublist]
            final_results.append(flat)

        return final_results


class DoubaoSeed_0615_EVAL:
    def __init__(self, *args, **kwargs) -> None:
        self.api_key = kwargs.get(
            "api_key",
            os.environ.get(
                "ARK_API_KEY",
                "xxx"
            )
        )
        self.model_name = kwargs.get(
            "model_name",
            "xxx"
        )
        self.thinking = kwargs.get(
            "thinking",
            {
                "type": "disabled",
            },
        )
        self.client = Ark(
            api_key=self.api_key
        )
    

    def request(self, messages: tuple, max_tokens: int, temperature: float, n: int, model: str, max_retries: int = 600, retry_delay: int = 60, base_delay: float = 1.5):
        id, ir, messages = messages
        messages = messages["messages"]
        responses = []
        for k in range(n):
            retries = 0
            while retries < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=1,
                        messages=messages,
                        thinking=self.thinking,
                    )
                    responses.append(completion.choices[0].message.content)
                    break

                except Exception as e:
                    retries += 1
                    exponential_delay = (2 ** retries) * base_delay
                    jitter = random.uniform(0, 100)
                    total_delay = exponential_delay + jitter

                    print(
                        f"[sample {k+1}] attempt {retries} failed: {e}. "
                    )
                    time.sleep(total_delay)
                    if retries == max_retries:
                        responses.append("ERROR")
        return id, ir, responses
    

    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict = None) -> list:
        if param_dict is None:
            param_dict = {}

        request_single_thread = functools.partial(
            self.request,
            max_tokens=param_dict.get("max_new_tokens", 16384),
            temperature=param_dict.get("temperature", None),
            n=param_dict.get("n", 1),
            model=self.model_name,
            max_retries=30,
            base_delay=1.5,
        )

        data_list_convert = convert_input_eval(
            data_list,
            min_pixels=param_dict.get("min_pixels", 1 * 1),
            max_pixels=param_dict.get("max_pixels", 4096 * 4096),
            is_encode_image=True,
        )
        
        results = []
        with ThreadPoolExecutor(max_workers=param_dict.get("max_workers", 10)) as executor:
            future_to_index = {
                executor.submit(request_single_thread, data_dict): i
                for i, data_dict in enumerate(data_list_convert)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Processing requests",
                unit="req"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx, ir = future_to_index[future]
                    results.append((idx, ir, f"Error: {e}"))

        from collections import defaultdict
        grouped = defaultdict(dict)
        for idx, ir, res in results:
            grouped[idx][ir] = res

        ir_counts = {idx: len(irs) for idx, irs in grouped.items()}
        if len(set(ir_counts.values())) != 1:
            raise ValueError(f"idx != ir: {ir_counts}")

        max_idx = max(grouped.keys())
        max_ir = max(ir for irs in grouped.values() for ir in irs.keys())

        final_results = []
        for i in range(data_list[0][0], data_list[-1][0] + 1):
            if i not in grouped:
                raise ValueError(f"don't have idx={i} results")
            inner = [grouped[i][ir] for ir in range(max_ir + 1)]
            flat = [item for sublist in inner for item in sublist]
            final_results.append(flat)

        return final_results


class DoubaoSeedBaseCls:
    def __init__(self, *args, **kwargs) -> None:
        self.api_key = kwargs.get(
            "api_key",
            os.environ.get(
                "ARK_API_KEY",
                "xxx"
            )
        )
        self.model_name = kwargs.get(
            "model_name",
            "xxx"
        )
        self.thinking = kwargs.get(
            "thinking",
            {
                "type": "disabled",
            },
        )
        self.client = Ark(
            api_key=self.api_key
        )

    def request(self, messages: tuple, max_tokens: int, temperature: float, n: int, model: str, max_retries: int = 10, retry_delay: int = 60, base_delay: float = 1.5) -> list:
        i, messages = messages
        responses = []

        for k in range(n):
            retries = 0
            while retries < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=1,
                        thinking=self.thinking,
                    )
                    responses.append(completion.choices[0].message.content)
                    break
                except Exception as e:
                    retries += 1
                    exponential_delay = (2 ** retries) * base_delay
                    jitter = random.uniform(0, 100)
                    total_delay = exponential_delay + jitter
                    print(
                        f"[sample {k+1}/{n}] attempt {retries} failed: {e}. "
                    )
                    time.sleep(total_delay)
                    if retries == max_retries:
                        responses.append("ERROR")
        return i, responses

    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict = None) -> list:
        if param_dict is None:
            param_dict = {}

        request_single_thread = functools.partial(
            self.request,
            max_tokens=param_dict.get("max_new_tokens", 16384),
            temperature=param_dict.get("temperature", None),
            n=param_dict.get("n", 1),
            model=self.model_name,
            max_retries=30,
            base_delay=1.5,
        )

        data_list_convert = convert_input(
            data_list,
            min_pixels=param_dict.get("min_pixels", 1 * 1),
            max_pixels=param_dict.get("max_pixels", 4096 * 4096),
            is_encode_image=True,
        )

        results = []
        with ThreadPoolExecutor(max_workers=param_dict.get("max_workers", 10)) as executor:
            future_to_index = {
                executor.submit(request_single_thread, (i, data_dict["messages"])): i
                for i, data_dict in enumerate(data_list_convert)
            }

            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Processing requests",
                unit="req"
            ):
                try:
                    result = future.result()
                    index = future_to_index[future]
                    results.append((index, result))
                except Exception as e:
                    index = future_to_index[future]
                    results.append((index, f"Error: {e}"))  

        final_results = [result for _, result in sorted(results)]

        return final_results


class DoubaoSeed16Flash_0715(DoubaoSeedBaseCls):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DoubaoSeed16_0611(DoubaoSeedBaseCls):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class DoubaoSeed16Thinking_0715(DoubaoSeedBaseCls):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thinking = None