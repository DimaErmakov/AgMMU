import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scoring_eval_pipeline.utils import ModelHandler

if __name__ == "__main__":
    handler = ModelHandler()
    
    # Test without image
    prompt_text = """The following question has this background information:
background info: had hail damage in June, other trees in the yard such as green ash, locust, maple, basswood and crabapple trees are not affected, no Asian beetles seen
species: birch tree
location: Hennepin County,Minnesota
time: 2017-08-19 14:50:54
Question: What insect is indicated by this image? Answer in 1-3 words."""
    print("=== Testing without image ===")
    try:
        response = handler.generate_response(
            system="You are a helpful AI assistant.",
            prompt=prompt_text,
            image_path=None
        )
        print("Response without image:", response)
    except Exception as e:
        print("Error without image:", e)

    # Test with image
    prompt_image = "What is shown in this image?"
    test_image_path = "/u/dermakov/AgMMU/images/copied_images/422159/422159_1.png"
    print("\n=== Testing with image ===")
    try:
        response = handler.generate_response(
            system="You are a helpful AI assistant.",
            prompt=prompt_text,
            image_path=test_image_path
        )
        print("Response with image:", response)
    except Exception as e:
        print("Error with image:", e) 