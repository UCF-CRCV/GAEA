#!/usr/bin/env python3
"""
Combined script to process evaluation results and calculate metrics.
This script combines the functionality of get_scores.py and calculate_metrics.py.
"""
import os
import json
import argparse
import pandas as pd
from glob import glob
from json_repair import loads as json_repair_loads

def process_response_file(response_file, bench_dict):
    """
    Process a response file to extract scores.
    
    Args:
        response_file: Path to the response file (.jsonl)
        bench_dict: Dictionary mapping unique_ids to benchmark data
        
    Returns:
        List of dictionaries containing processed scores
    """
    print(f"Processing {response_file}")
    processed_data = []
    
    # Read response file
    with open(response_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        try:
            item = json.loads(line)
            unique_id = item['custom_id']
            # Extract the numeric part after the last underscore
            bench_id = str(unique_id).split('_')[-1]
            
            # Get response content
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            
            # Skip if benchmark data doesn't exist for this ID
            if bench_id not in bench_dict:
                continue
                
            # Get benchmark data
            sample = bench_dict[bench_id]
            question_type = sample['question_type']
            
            # Parse score based on question type
            score = 0
            if question_type == 'SVQA':
                try:
                    score_dict = json_repair_loads(content)
                    score = sum(int(v) for v in score_dict.values()) / 2
                except:
                    print(f"Error parsing SVQA score for {unique_id}")
                    continue
            else:
                try:
                    score = int(content)
                except:
                    print(f"Error parsing score for {unique_id}")
                    continue
            
            # Create result entry
            processed_data.append({
                'image': sample['image'],
                'unique_id': sample['unique_id'],
                'question_type': question_type,
                'score': score
            })
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
    
    return processed_data

def calculate_metrics(processed_data, model_name):
    """
    Calculate metrics from processed data.
    
    Args:
        processed_data: List of dictionaries containing processed scores
        model_name: Name of the model
        
    Returns:
        Dictionary containing calculated metrics
    """
    df = pd.DataFrame(processed_data)
    
    # Adjust model name based on specific conditions
    if model_name in ['ablation1', 'ablation2']:
        model_name = 'GAEA-' + model_name + "+"
    if model_name == 'baseline':
        model_name = model_name + "*"
    
    # Extract scores by question type
    long_qa = df[df['question_type'] == 'LVQA']['score'].astype(int).tolist() if 'LVQA' in df['question_type'].values else []
    short_qa = df[df['question_type'] == 'SVQA']['score'].astype(int).tolist() if 'SVQA' in df['question_type'].values else []
    mcq_qa = df[df['question_type'] == 'MCQ']['score'].astype(int).tolist() if 'MCQ' in df['question_type'].values else []
    tf_qa = df[df['question_type'] == 'TF']['score'].astype(int).tolist() if 'TF' in df['question_type'].values else []
    
    # Calculate averages (normalized to a 0-10 scale)
    normalize_val = 10
    lvqa_avg = round(sum(long_qa) / len(long_qa) * normalize_val, 2) if len(long_qa) > 0 else 0
    svqa_avg = round(sum(short_qa) / len(short_qa) * normalize_val, 2) if len(short_qa) > 0 else 0
    mcq_avg = round(sum(mcq_qa) / len(mcq_qa) * normalize_val, 2) if len(mcq_qa) > 0 else 0
    tf_avg = round(sum(tf_qa) / len(tf_qa) * normalize_val, 2) if len(tf_qa) > 0 else 0
    total_avg = round((lvqa_avg + svqa_avg + mcq_avg + tf_avg) / 4, 2) if len(df) > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'Model': model_name,
        'Long QA Average': lvqa_avg,
        'Short QA Average': svqa_avg,
        'MCQ QA Average': mcq_avg,
        'TF QA Average': tf_avg,
        'Total Average': total_avg,
        'Long QA Count': len(long_qa),
        'Short QA Count': len(short_qa),
        'MCQ QA Count': len(mcq_qa),
        'TF QA Count': len(tf_qa),
        'Total Files': len(df)
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing metrics
    """
    model_name = metrics['Model']
    
    hashtag_print_stmt = '############################################################################'
    title_print_stmt = f"#############################Results for {model_name}#############################" \
                      if model_name == "GAEA" else f"Results for {model_name}"
    
    if model_name in ["GPT", "Gemini"]:
        print(hashtag_print_stmt)
    print(hashtag_print_stmt)
    print(title_print_stmt)
    
    print(f'Long QA Average: {metrics["Long QA Average"]} --> There are {metrics["Long QA Count"]} long questions')
    print(f'Short QA Average: {metrics["Short QA Average"]} --> There are {metrics["Short QA Count"]} short questions')
    print(f'MCQ QA Average: {metrics["MCQ QA Average"]} --> There are {metrics["MCQ QA Count"]} MCQs')
    print(f'TF QA Average: {metrics["TF QA Average"]} --> There are {metrics["TF QA Count"]} TFs')
    print(f'Total Average: {metrics["Total Average"]}')
    print(f'Total Files: {metrics["Total Files"]}\n')

def main():
    parser = argparse.ArgumentParser(description="Process evaluation results and calculate metrics")
    parser.add_argument("--response_dir", type=str, required=True, 
                        help="Directory containing response JSONL files")
    parser.add_argument("--benchmark_file", type=str, required=True,
                        help="Path to the benchmark JSON file")
    parser.add_argument("--output_csv", type=str, default="scores.csv",
                        help="Path to output CSV file")
    
    args = parser.parse_args()
    

    # Load benchmark data
    with open(args.benchmark_file, 'r') as f:
        bench_data = json.load(f)
    bench_dict = {str(item['unique_id']): item for item in bench_data}
    
    # Find response files
    response_files = glob(f"{args.response_dir}/*.jsonl")
    print(f"Found {len(response_files)} response files")
    
    all_metrics = []
    
    # Process each response file
    for response_file in response_files:
        model_name = os.path.basename(response_file).replace('.jsonl', '')
        
        # Process responses and calculate metrics
        processed_data = process_response_file(response_file, bench_dict)
        metrics = calculate_metrics(processed_data, model_name)
        
        # Print metrics
        print_metrics(metrics)
        
        # Add to all metrics
        all_metrics.append(metrics)
    
    # Create and save dataframe
    results_df = pd.DataFrame(all_metrics)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\nAll Results:")
    print(results_df.to_string(index=False))
    
    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main() 