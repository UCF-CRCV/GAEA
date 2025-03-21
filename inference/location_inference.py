"""
This script performs location inference on images using the Qwen-2.5-VL model.
It supports both direct image directory processing and CSV-based dataset processing
for specific datasets (Dollar Street, YouTube, GeoDE).
"""

# import requests
# import copy
# import torch
import json
import time
import os
from tqdm import tqdm 
import argparse
# import numpy as np
# from PIL import Image
import warnings
from glob import glob
from accelerate import Accelerator
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Inference example for Qwen-2.5-VL")
    parser.add_argument("--image_dir", type=str, default="", help="Path to the directory containing the images")
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Name of the pretrained model")
    parser.add_argument("--outpath", type=str, help="Path for the output JSON file")
    parser.add_argument("--device", type=str, default="cuda", help="Cuda or CPU")
    parser.add_argument("--temperature", type=float, default=0.0, help='If 0, sets model as deterministic')
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--csv_data', type=str, default="", help="Path to CSV file containing dataset information (for Dollar Street, YouTube, or GeoDE)")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained model weights")
    return parser.parse_args()


args = parse_args()
pretrained = args.pretrained
device = args.device
model_base = args.model_base

pretrained = model_base if pretrained == "" else pretrained
print(f'Inference on model: {pretrained}')

min_pixels = 224*224
max_pixels = 2048*2048
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", low_cpu_mem_usage=True, attn_implementation='flash_attention_2', device_map="auto")
# processor = AutoProcessor.from_pretrained(model_base)
processor = AutoProcessor.from_pretrained(pretrained, padding_side="right", min_pixels=min_pixels, max_pixels=max_pixels)
model.eval()

def fixed_location_prompt():
    return 'As a geography and tourism expert, analyze the image to determine its exact location. Utilize your extensive knowledge of geography, terrain, landscapes, flora, fauna, infrastructure, and recognizable landmarks to identify the city and country where the image was taken. Question: '

def process_video_question(image_path, question, temperature):
    """
    Process a single image and generate location prediction.
    
    Args:
        image_path (str): Path to the image file
        question (str): The question to ask about the image
        temperature (float): Temperature for generation (0.0 for deterministic)
    
    Returns:
        list: Model's predicted answer
    """
    user_prompt = f"{fixed_location_prompt()}{question}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image_path, 
                    'min_pixel': min_pixels,
                    'max_pixel': max_pixels
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    if temperature == 0.0:
        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=20, temperature=temperature)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    predicted_answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return predicted_answer

def process_json_file(files, results, temperature, full_output_path):
    """
    Process a list of image files and generate location predictions.
    
    Args:
        files (list): List of image file paths
        results (list): List to store results
        temperature (float): Temperature for generation
        full_output_path (str): Path to save results
    
    Returns:
        tuple: (results list, total processing time)
    """
    start_time = time.time()
    already_seen = set([result['filename'] for result in results])
    
    for i, file in enumerate(tqdm(files, desc="Processing images", unit="image")):
        question = "Where is this image taken? Respond with only the city and country."

        # Determine filename based on dataset type
        if "Dollar Street" in files[0]:
            video_filename = file.split("/")[-1]
        elif "YouTube" in files[0]:
            video_filename = "/".join(file.split("/")[-2:])
        elif "GeoDE" in files[0]:
            video_filename = "/".join(file.split("/")[-3:])
        elif "yfcc26k" in files[0]:
            video_filename = "/".join(file.split("/")[-3:])
        else:
            video_filename = file.split("/")[-1]

        if video_filename in already_seen:
            continue

        if not os.path.exists(file):
            model_answer = f"Error: Image file not found at {file}"
        else:
            model_answer = process_video_question(file, question, temperature)

        results.append({
            'filename': video_filename,
            'question': question,
            'predicted_answer': model_answer
        })

        # Save intermediate results every 100 images
        if i % 100 == 0:
            with open(full_output_path, 'w') as file:
                json.dump(results, file, indent=4)

    end_time = time.time()
    total_time = end_time - start_time

    return results, total_time

def get_files_from_csv(image_dir, csv_data):
    """
    Get list of image files from CSV data for specific datasets.
    
    Args:
        image_dir (str): Base directory containing images
        csv_data (str): Path to CSV file containing dataset information
    
    Returns:
        list: List of image file paths
    """
    evaluation_data = pd.read_csv(csv_data)
    
    if "Dollar Street" in image_dir:
        files = evaluation_data['id'].apply(lambda x: os.path.join(image_dir, x + '.jpg')).tolist()
    elif "YouTube" in image_dir:
        files = evaluation_data['IMG_ID'].apply(lambda x: os.path.join(image_dir, x)).tolist()
    elif "GeoDE" in image_dir:
        files = evaluation_data['file_path'].apply(lambda x: os.path.join(image_dir, x)).tolist()
    else:
        raise ValueError(f"Unsupported dataset type in directory: {image_dir}")
    
    return files

def main():
    image_dir = args.image_dir
    outpath = args.outpath
    save_dir = args.save_dir
    temperature = args.temperature
    csv_data = args.csv_data

    print("Starting image processing...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get list of files to process
    if csv_data:
        # Process files from CSV for specific datasets
        files = get_files_from_csv(image_dir, csv_data)
    else:
        # Process all images in directory
        if "yfcc26k" in image_dir:
            files = glob(image_dir + "/**/**/*.jpg")
        else:
            files = glob(image_dir + "/*.jpg")
    
    print(save_dir)
    full_output_path = os.path.join(save_dir, outpath)

    # Load existing results if available
    if os.path.exists(full_output_path):
        results = json.load(open(full_output_path))
    else:
        results = []

    # Process images and generate predictions
    results, total_time = process_json_file(files, results, temperature, full_output_path)

    # Save final results
    with open(full_output_path, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"\nProcessing completed in {total_time:.2f} seconds.")
    print(f"Results saved to {full_output_path}")

    # Print summary
    print(f"\nProcessed {len(results)} Images.")
    print(f"Average time per Image: {total_time / len(results):.2f} seconds.")

if __name__ == "__main__":
    main()

