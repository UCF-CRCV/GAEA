#!/usr/bin/env python3
"""
Script to create evaluation batches for city prediction accuracy.
Uses Azure OpenAI to evaluate model predictions against ground truth.
"""
import json
import csv
import os
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

def get_client():
    """Initialize and return the Azure OpenAI client."""
    load_dotenv()
    
    # Get configuration from environment variables, with no defaults for sensitive data
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    
    if not endpoint or not api_key:
        raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in environment variables or .env file")
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

def load_predictions(pred_path):
    """Load model predictions from a JSON file."""
    with open(pred_path, 'r') as f:
        results = json.load(f)
    
    predictions = {}
    for i in results:
        # Extract city (first part of the location)
        predictions[i['filename']] = i['predicted_answer'].split(',')[0]
    
    return predictions

def load_ground_truth(gt_path, benchmark, city_index):
    """Load ground truth data from a CSV file."""
    gts = {}
    
    with open(gt_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader, None)  # Skip header
        for line in csvreader:
            if benchmark == 'yfcc4k':
                key = line[0] + '.jpg'
            else:    
                key = line[0]
            gts[key] = line[city_index]
    
    return gts

def get_city_index(benchmark):
    """Get the index of city information in the ground truth CSV."""
    if benchmark == 'geode':
        return 3
    elif benchmark == 'cityguessr':
        return 1
    else:
        return 1  # Default to 1 if unknown

def create_evaluation_batches(predictions, ground_truth, benchmark, model_name, deployment_name):
    """Create evaluation batches for the Azure OpenAI API."""
    batch = []
    
    for filename in tqdm(ground_truth):
        if filename not in predictions:
            print(f"Warning: {filename} in ground truth but not in predictions. Skipping.")
            continue
            
        pred = predictions[filename]
        gt = ground_truth[filename]
        
        prompt_eval = (
            f"Evaluate this considering the geographical context.:\n\n"
            f"If the prediction is within the ground truth, it is correct.\n"
            f"If it is a match give a score of 1, otherwise, give a score of 0.\n"
            f"Prediction: {pred}\n"
            f"Ground Truth: {gt}\n"
            f"Just return the score without any additional commentary."
        )
        
        bat = {
            "custom_id": f"{filename}-benchmark-{benchmark}-{model_name}",
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": deployment_name, 
                "messages": [
                    {"role": "system", "content": "You are a helpful Assistant. Provide helpful response to the user's question."},
                    {"role": "user", "content": prompt_eval}
                ],
                "max_tokens": 1000
            }
        }
        batch.append(bat)
    
    return batch

def save_batch_file(batch, output_path, benchmark, model_folder, eval_type="city"):
    """Save the evaluation batch to a JSONL file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for entry in batch:
            f.write(json.dumps(entry))
            f.write('\n')
    
    print(f"Saved batch file to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create evaluation batches for city predictions")
    parser.add_argument("--pred_folder", type=str, required=True, 
                        help="Path to folder containing prediction JSON files")
    parser.add_argument("--gt_folder", type=str, required=True,
                        help="Path to folder containing ground truth CSV files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to folder where batch files will be saved")
    parser.add_argument("--model_name", type=str, default="GAEA",
                        help="Name of the model being evaluated")
    parser.add_argument("--deployment_name", type=str, default="gpt-4o-mini",
                        help="Azure OpenAI deployment name to use for evaluation")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="List of benchmarks to exclude")
    args = parser.parse_args()
    
    # Set up Azure OpenAI client
    client = get_client()
    
    # Get prediction files
    pred_files = glob(os.path.join(args.pred_folder, "*.json"))
    
    # Remove excluded benchmarks
    for excluded in args.exclude:
        excluded_path = os.path.join(args.pred_folder, f"{excluded}.json")
        if excluded_path in pred_files:
            pred_files.remove(excluded_path)
    
    # Process each prediction file
    for pred_path in pred_files:
        benchmark = Path(pred_path).stem
        print(f"Processing benchmark: {benchmark}")
        
        # Set paths
        gt_path = os.path.join(args.gt_folder, f"{benchmark}_citygt.csv")
        output_path = os.path.join(args.output_folder, f"{benchmark}_{Path(args.pred_folder).name}_city.jsonl")
        
        # Check if ground truth file exists
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file {gt_path} not found. Skipping benchmark {benchmark}.")
            continue
        
        # Get city index based on benchmark
        city_index = get_city_index(benchmark)
        
        # Load predictions and ground truth
        predictions = load_predictions(pred_path)
        ground_truth = load_ground_truth(gt_path, benchmark, city_index)
        
        # Create evaluation batches
        batch = create_evaluation_batches(predictions, ground_truth, benchmark, args.model_name, args.deployment_name)
        
        # Save batch file
        save_batch_file(batch, output_path, benchmark, Path(args.pred_folder).name)

if __name__ == "__main__":
    main()
