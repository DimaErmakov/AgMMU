#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
import sys

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from AgMMU.scoring_eval_pipeline.utils import ModelHandler

# Initialize model once globally
model_handler = None

def format_options(q):
    options = q["options"]
    st = ""
    for option, letter in zip(options, ["A.", "B.", "C.", "D."]):
        if option == q["answer"]:
            q["letter"] = letter
        st += f"{letter} {option}\n"
    return st


def run_llms(prompt, img, q):
    # print(f"Running LLM with image: {img}")
    # print(f"Prompt: {prompt}")
    
    system = "You are a helpful AI assistant."
    try:
        global model_handler
        if model_handler is None:
            model_handler = ModelHandler()  # Initialize Qwen2-VL model only once
            print("Model initialized successfully")

        response = model_handler.generate_response(system, prompt, img)
        print(f"Got response: {response}")
        q['answer'] = response
    except Exception as e:
        print(f"Error in run_llms: {str(e)}")
        # exit(1)
        raise


def add_item_to_json(file_path, new_item):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = []

    if isinstance(new_item, list):
        data.extend(new_item)
    else:
        data.append(new_item)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def eval_data(data, llm_map, output, image_dir):
    seen_ids = set()
    if os.path.exists(output):
        with open(output, 'r') as file:
            existing = json.load(file)
        seen_ids = {item['faq-id'] for item in existing}

    for q in data:
        if q['faq-id'] in seen_ids:
            print(f"Skipping {q['faq-id']} because it already exists")
            continue

        try:
            q['llm_answers'] = llm_map
            for llm in q['llm_answers']:
                prefix = q['agmmu_question'].get('question_background', '')
                qtype = q['qtype']
                faq_id = q['faq-id']
                
                # Try different image paths
                possible_paths = [
                    os.path.join(image_dir, str(faq_id), f"{faq_id}_1.jpg"),
                    os.path.join(image_dir, str(faq_id), f"{faq_id}_1.png"),
                ]
                
                img_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        # Test if the image can be opened successfully
                        try:
                            from PIL import Image
                            with Image.open(path) as img:
                                img.verify()  # Verify the image can be opened
                            img_path = path
                            break
                        except Exception as e:
                            print(f"Warning: Corrupted image {path}: {str(e)}")
                            continue
                
                if img_path is None:
                    print(f"Skipping {faq_id}: No valid image found")
                    continue

                if 'mcq' in llm:
                    prompt = (
                        f"{prefix}{q['agmmu_question']['question']}\n"
                        f"Options:\n{format_options(q['agmmu_question'])}\n"
                        "Choose the letter corresponding with the correct answer. Only output the single letter."
                    )
                else:
                    if qtype in ['disease/issue identification', 'insect/pest', 'species']:
                        prompt = f"Question: {q['agmmu_question']['question']} Answer in 1-3 words."
                    elif qtype == 'management instructions':
                        prompt = "What is the recommended management strategy for the issue seen in this image?\nBe descriptive."
                    elif qtype == 'symptom/visual description':
                        prompt = "What visual features can be seen in this image?\nBe descriptive."
                    else:
                        print("Unknown qtype:", qtype)
                        continue
                    prompt = prefix + prompt
                
                run_llms(prompt, img_path, q['llm_answers'][llm])
                add_item_to_json(output, q)

        except Exception as e:
            print("Error:", e)
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate visual symptom questions without RAG."
    )
    parser.add_argument("--data_path", required=True, help="Path to main data JSON.")
    parser.add_argument(
        "--output_path", required=True, help="Path to save output JSON."
    )
    parser.add_argument(
        "--image_dir", required=True, help="Directory with image files."
    )
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    random.shuffle(data)
    print(f"Loaded {len(data)} questions")
    eval_data(
        data=data,
        llm_map={"qwen2.5-vl-oeq": {}, "qwen2.5-vl-mcq": {}},  # Updated model names
        output=args.output_path,
        image_dir=args.image_dir,
    )


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main() 
