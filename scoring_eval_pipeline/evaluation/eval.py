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

from scoring_eval_pipeline.utils import ModelHandler

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


def run_llms(prompt, img, q, top_k_10=None):
    # print(f"Running LLM with image: {img}")
    # print(f"Prompt: {prompt}")
    
    # Feel free to change the prefix to your liking but I think this is a good one
    prefix = """CRITICAL INSTRUCTIONS: 
1. The LAST image is the MAIN QUESTION image that you must analyze and answer about
2. The preceding images and text are RETRIEVED EXAMPLES that provide relevant context to help you answer the main question
3. Focus your analysis primarily on the LAST image, but use the retrieved examples to inform your understanding
4. Your answer should directly address the question about the LAST image, using insights from the retrieved examples when relevant
5. You should not use the retrieved examples to answer the question, you should use them to inform your understanding of the LAST image
6. Carefully examine the retrieved examples to understand patterns, symptoms, or characteristics that may apply to the main question image
7. Use the retrieved examples as reference material to enhance your analysis of the main question image
8. Provide a comprehensive answer that demonstrates understanding of both the specific question and the broader context from the retrieved examples

MAIN QUESTION (about the last image): """
    
    prompt = prefix + prompt
    
    system = "You are a helpful AI assistant."
    try:
        global model_handler
        if model_handler is None:
            model_handler = ModelHandler()  # Initialize Qwen2-VL model only once
            print("Model initialized successfully")

        response = model_handler.generate_response(system, prompt, img, top_k_10)
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


def eval_data(data, llm_map, output, image_dir, top_k_10_path):
    # This method now has the top 10 retrieval results loaded
    print(f"Loading top 10 retrieval results from {top_k_10_path}")
    with open(top_k_10_path, "r") as f:
        top_k_10_list = json.load(f)
    top_k_10_dict = {str(item["faq-id"]): item for item in top_k_10_list}
    # print(top_k_10_dict)
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
                img_path = os.path.join(image_dir, str(faq_id), f"{faq_id}_1.png")
                # img_path = os.path.join(image_dir, "copied_images", str(faq_id), f"{faq_id}_1.png")
                if not os.path.exists(img_path):
                    # print(f"Skipping {faq_id}: Image not found at {img_path}")
                    img_path = os.path.join(image_dir, "copied_images", str(faq_id), f"{faq_id}_1.png")
                    if not os.path.exists(img_path):
                        # print(f"Skipping {faq_id}: Image not found at {img_path}")
                        continue

                top_k_10 = top_k_10_dict.get(str(faq_id))
                if top_k_10:
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
                    run_llms(prompt, img_path, q['llm_answers'][llm], top_k_10)
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
    # parser.add_argument(
    #     "--top_k_10", required=True, help="Path to top 10 retrieval results."
    # )
    top_k_10 = ""
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
        top_k_10_path=args.top_k_10,
    )


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()
