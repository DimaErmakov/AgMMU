#!/usr/bin/env python3
"""
Script to add MBEIR format fields to each item in the JSON file.
"""

import json
import os

def add_mbeir_fields_to_json(input_file_path, output_file_path=None):


    
    
    print(f"Reading JSON file: {input_file_path}")
    
    # Read the JSON file
    with open(input_file_path) as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items in the JSON file")
    
    # Add the required fields to each item
    for i, item in enumerate(data):
        # Add the new fields
        item["query_modality"] = "image,text"
        item["query_src_content"] = None
        item["pos_cand_list"] = []
        item["neg_cand_list"] = []
        item["task_id"] = 0
        
        # Print progress every 1000 items
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} items...")
    
    print(f"Writing modified JSON to: {output_file_path}")
    
    # Write the modified data back to the file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print("Successfully added MBEIR fields to all items!")
    
    # Verify the changes by reading a sample
    print("\nVerifying changes...")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    if sample_data:
        sample_item = sample_data[0]
        print("Sample item with new fields:")
        for field in ["query_modality", "query_src_content", "pos_cand_list", "neg_cand_list", "task_id"]:
            print(f"  {field}: {sample_item.get(field)}")

def main():
    """Main function to run the script."""
    input_file = "/u/dermakov/AgMMU/data/6k_evalset_wbg_add_on.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    
    print("Starting to add MBEIR fields to JSON file...")
    add_mbeir_fields_to_json(input_file, input_file)
    print("Done!")

if __name__ == "__main__":
    main() 