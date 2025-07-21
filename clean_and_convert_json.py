#!/usr/bin/env python3
"""
Script to clean and convert JSON data from AgMMU format to MBEIR format.
"""

import json
import os
import re
from typing import Dict, Any, List

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags and normalizing whitespace.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_image_path(attachments: Dict[str, str]) -> str:
    """
    Extract the first image path from attachments.
    
    Args:
        attachments (Dict[str, str]): Dictionary of attachments
        
    Returns:
        str: Image path or empty string
    """
    if not attachments:
        return ""
    
    # Get the first attachment (usually the main image)
    for key, value in attachments.items():
        if value and isinstance(value, str):
            return value
    
    return ""

def convert_to_mbeir_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an AgMMU item to MBEIR format.
    
    Args:
        item (Dict[str, Any]): Original AgMMU item
        
    Returns:
        Dict[str, Any]: MBEIR format item
    """
    # Extract and clean the question text
    question = clean_text(item.get("agmmu_question", {}).get("question"))
    
    # Extract image path from attachments
    image_path = extract_image_path(item.get('attachments', {}))
    
    # Create unique ID for this item
    faq_id = item.get('faq-id', 0)
    qid = f"1:{faq_id}"  # Using dataset ID 1 for AgMMU
    
    # Create the MBEIR format item
    mbeir_item = {
        "qid": qid,
        "query_txt": question,
        "query_img_path": image_path if image_path else None,
        "query_modality": "image,text" if image_path else "text",
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
        "task_id": 0,
        "faq-id": item.get("faq-id"),
        "title": item.get("title"),
        "created": item.get("created"),
        "updated": item.get("updated"),
        "state": item.get("state"),
        "county": item.get("county"),
        "tags": item.get("tags", []),
        "attachments": item.get("attachments", {}),
        "question": item.get("question"),
        "answer": item.get("answer", []),
        "species": item.get("species"),
        "category": item.get("category"),
        "qa_information": item.get("qa_information", {}),
        "agmmu_question": item.get("agmmu_question", {}),
        "qtype": item.get("qtype")
    }
    
    return mbeir_item

def clean_and_convert_json(source_path: str, target_path: str):
    """
    Clean and convert JSON data from source to target format.
    
    Args:
        source_path (str): Path to source JSON file
        target_path (str): Path to target JSONL file
    """
    print(f"Reading source file: {source_path}")
    
    # Read the source JSON file
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items in source file")
    
    # Create target directory if it doesn't exist
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    
    # Convert and write items
    converted_count = 0
    skipped_count = 0
    
    with open(target_path, 'w', encoding='utf-8') as f:
        for item in data:
            try:
                # Convert to MBEIR format
                mbeir_item = convert_to_mbeir_format(item)
                
                # Write as JSONL (one JSON object per line)
                f.write(json.dumps(mbeir_item, ensure_ascii=False) + '\n')
                converted_count += 1
                
                # Print progress every 100 items
                if converted_count % 100 == 0:
                    print(f"Processed {converted_count} items...")
                    
            except Exception as e:
                print(f"Error processing item {item.get('faq-id', 'unknown')}: {e}")
                skipped_count += 1
                continue
    
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted_count} items")
    print(f"Skipped: {skipped_count} items")
    print(f"Output saved to: {target_path}")
    
    # Verify the output file
    print("\nVerifying output file...")
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    json.loads(line)  # Test JSON parsing
                    line_count += 1
                    
                    # Show first few items as examples
                    if line_count <= 3:
                        item = json.loads(line)
                        print(f"Sample item {line_count}:")
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                        print()
            
            print(f"Successfully validated {line_count} JSON lines")
            
    except Exception as e:
        print(f"Error validating output file: {e}")

def main():
    """Main function to run the conversion."""
    # source_file = "/work/nvme/bdbf/dermakov/agmmu/AgMMU_v1/query/test/mbeir_sample_test.jsonl"
    source_file = "/u/dermakov/AgMMU/data/6k_evalset_wbg_add_on.jsonl"
    target_file = "/work/nvme/bdbf/dermakov/agmmu/AgMMU_v1/query/test/mbeir_sample_test.jsonl"
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} does not exist!")
        return
    
    print("Starting JSON conversion and cleaning...")
    clean_and_convert_json(source_file, target_file)
    print("Done!")

if __name__ == "__main__":
    main() 