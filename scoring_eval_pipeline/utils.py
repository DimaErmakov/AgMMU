import json
import os
import requests
from openai import APIError, RateLimitError
from io import BytesIO
from PIL import Image
import re
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup as bs
import time
import base64
import random
import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import subprocess

def exponential_backoff(func, *args, max_retries=100, delay=1):
    cnt = 0
    while cnt < max_retries:
        try:
            return func(*args)
        except RateLimitError as e:
            print(f"Rate limit reached: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
            cnt += 1
        except Exception as e:
            print(e)
            break
    raise Exception(f"Failed to execute {func.__name__} after {max_retries} retries.")


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_file)
    return image


def add_item_to_json(file_path, new_item):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path) as file:
            data = json.load(file)
    else:
        with open(file_path, "w") as json_file:
            json.dump([], json_file)
        data = []
    if isinstance(new_item, list):
        data.extend(new_item)
    else:
        data.append(new_item)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


import re
import json

def clean_response(response):
    # Remove markdown code block markers
    response = re.sub(r"^```json|```$", "", response, flags=re.MULTILINE).strip()

    # Try to extract the first JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print("Raw extracted JSON string:", json_str)
            raise
    else:
        print("No JSON object found in response!")
        print("Raw response:", response)
        raise ValueError("No JSON object found in response")
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def chat_gpt(system, prompt, image_path=None):
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini")
    # get your own api key
    os.environ["OPENAI_API_KEY"] = config["DEFAULT"]["OPENAI_API_KEY"]
    client = OpenAI()

    if image_path is None:
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    else:
        base64_image = encode_image(image_path)
        message = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

    response = client.chat.completions.create(
        model="gpt-4.1", messages=message, temperature=0.3
    )
    print(response.choices[0].message.content.strip())
    # exit(1) # if you want to test the code, uncomment this
    return response.choices[0].message.content.strip()

def create_message(system, prompt, image_path):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    if system:
        message.insert(0, {"role": "system", "content": system})

    if image_path:
        base64_image = encode_image(image_path)
        message[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    return message


# def resize_image(image_path, target_size=(224, 224)):
#     # Use this if you want to resize the image since images are computationally expensive
#     try:
#         img = Image.open(image_path)
#         img = img.convert("RGB")
#         img = img.resize(target_size, Image.BILINEAR)
#         return img
#     except Exception as e:
#         print(f"Error resizing image {image_path}: {e}")


class ModelHandler:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_name = model_name
        print(f"Loading model {model_name}...")
        import os
        # Initialize model with flash attention
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded successfully")
        
        # Initialize processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("Processor loaded successfully")

    def generate_response(self, system: str, prompt: str, image_path=None, top_k_10=None):
        # Not going to lie, this code is a bit of a mess but it works
        print(f"Generating response for {self.model_name} with prompt: {prompt} and image: {image_path}")
        if image_path:
            # resized_image = resize_image(image_path, target_size=(224, 224))
            content = []
            # Add the main image first if it exists
            
            # Add the top 10 retrieval images and their corresponding text
            candidates = top_k_10.get("candidates", [])

            if isinstance(candidates, list) and candidates:
                for idx, cand in enumerate(candidates, 1):
                    if isinstance(cand, dict):
                        img_path = cand.get("img")
                        # if img_path and os.path.exists(img_path):
                            # resized_image = resize_image(img_path, target_size=(224, 224))
                        print(f"img_path: {img_path}")
                        if img_path and os.path.exists(img_path):
                            content.append({"type": "image", "image": img_path})
                            content.append({
                                "type": "text",
                                "text": f"Retrieved Example {idx}: {cand.get('txt', '')}"
                            })
                        else:
                            print(f"Warning: top_k_10 image path does not exist: {img_path}")
                    else:
                        print(f"Warning: candidate is not a dictionary: {cand}")
            else:
                print("Warning: candidates list is empty or not valid")

            if os.path.exists(image_path):
                content.append({"type": "image", "image": image_path})
            else:
                print(f"Warning: main image path does not exist: {image_path}")
            # Add the prompt text
            content.append({"type": "text", "text": prompt})
            payload = {
                "model": "qwen2.5vl:7b",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
        else:
            content = []
            candidates = top_k_10.get("candidates", [])

            if isinstance(candidates, list) and candidates:
                for idx, cand in enumerate(candidates, 1):
                    if isinstance(cand, dict):
                        img_path = cand.get("img")
                        # if img_path and os.path.exists(img_path):
                            # resized_image = resize_image(img_path, target_size=(224, 224))
                        print(f"img_path: {img_path}")
                        if img_path and os.path.exists(img_path):
                            content.append({"type": "image", "image": img_path})
                            content.append({
                                "type": "text",
                                "text": f"Retrieved Example {idx}: {cand.get('txt', '')}"
                            })
                        else:
                            print(f"Warning: top_k_10 image path does not exist: {img_path}")
                    else:
                        print(f"Warning: candidate is not a dictionary: {cand}")
            else:
                print("Warning: candidates list is empty or not valid")


            # Add the prompt text
            content.append({"type": "text", "text": prompt})
            payload = {
                "model": "qwen2.5vl:7b",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
        print(payload["messages"])

        # Prepare text input using the full messages structure (with both images and text)
        text = self.processor.apply_chat_template(
            payload["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision information if image is present
        if image_path:
            # Extract only image entries from the content
            image_entries = []
            for message in payload["messages"]:
                for entry in message["content"]:
                    if entry.get("type") == "image":
                        image_entries.append(entry)
            # Now process only the image entries
            image_inputs, _ = process_vision_info([{"role": "user", "content": image_entries}])
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )

        # Move inputs to the same device as the model
        inputs = inputs.to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Trim the generated IDs to only include new tokens
        generated_ids_trimmed = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the response
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(response.strip())
        return response.strip()
