#!/usr/bin/env python3
"""
Script to retrieve results from Azure OpenAI batch jobs for city-country evaluations.
Continues polling until all jobs are complete.
"""
import os
import json
import time
import datetime
import argparse
from glob import glob
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

def get_client():
    """Initialize and return the Azure OpenAI client."""
    load_dotenv()
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def retrieve_batch_results(client, batch_id):
    """
    Retrieve results from a batch job.

    Args:
        client: Azure OpenAI client
        batch_id: Batch ID to retrieve results for

    Returns:
        tuple: (batch_response, file_content, status)
    """
    batch_response = client.batches.retrieve(batch_id)
    status = batch_response.status
    print(f"{datetime.datetime.now()} Batch ID: {batch_id}, Status: {status}")

    output_file_id = batch_response.output_file_id
    if not output_file_id:
        output_file_id = batch_response.error_file_id

    file_content = None
    if output_file_id:
        file_response = client.files.content(output_file_id)
        print(f'Batch ID: {batch_id} for input file ID: {batch_response.input_file_id}')
        file_content = file_response.text
    
    return batch_response, file_content, status

def process_batch_results(file_content, output_file, eval_type):
    """
    Process batch results and save to output file.
    
    Args:
        file_content: Content of the batch results file
        output_file: Path to save the processed results
        eval_type: Evaluation type (city or country)
        
    Returns:
        processed_results: List of processed results
    """
    raw_responses = file_content.strip().split('\n')
    
    final_outputs = []
    for raw_response in raw_responses:
        json_response = json.loads(raw_response)
        
        try:
            # Extract custom ID parts
            model_name = str(json_response["custom_id"]).split('-')[-1] 
            custom_id = str(json_response["custom_id"]).split('-')[0]
            
            # Extract content
            output = json_response["response"]["body"]["choices"][0]["message"]["content"]
            
            final_outputs.append({
                "custom_id": custom_id,
                "output": output
            })
        except Exception as e:
            print(f"Error processing response: {e}")
            continue
    
    print(f"Processed {len(final_outputs)} responses")
    
    # Sort by custom ID
    sorted_final_outputs = sorted(final_outputs, key=lambda item: item["custom_id"])
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(sorted_final_outputs, f, indent=2)
    
    print(f"Saved results to {output_file}")
    
    return sorted_final_outputs

def process_batch_jobs(client, batch_ids, output_dir, eval_type, poll_interval=60, max_retries=None):
    """
    Process batch jobs and retrieve results.
    Continues polling until all jobs are complete.

    Args:
        client: Azure OpenAI client
        batch_ids: List of batch IDs to process
        output_dir: Directory to save results
        eval_type: Evaluation type (city or country)
        poll_interval: Time in seconds between polls
        max_retries: Maximum number of retries (None for unlimited)

    Returns:
        bool: True if all jobs completed successfully, False otherwise
    """
    # Dictionary to track completion status of each batch
    batch_status = {batch_id: {'complete': False, 'status': None} for batch_id in batch_ids}
    retry_count = 0
    
    # Continue polling until all batches are complete or max retries reached
    while not all(status['complete'] for status in batch_status.values()):
        any_progress = False
        
        for batch_id in batch_ids:
            # Skip already completed batches
            if batch_status[batch_id]['complete']:
                continue
                
            batch_response, file_content, status = retrieve_batch_results(client, batch_id)
            
            # Update status
            batch_status[batch_id]['status'] = status
            
            # Check if complete
            if status in ['completed', 'cancelled', 'expired', 'failed']:
                batch_status[batch_id]['complete'] = True
                any_progress = True
                
                if status == 'completed' and file_content:
                    # Process and save results
                    model_name = batch_response.input_file_id.split('_')[-1] if '_' in batch_response.input_file_id else 'unknown'
                    output_file = os.path.join(output_dir, f"{model_name}_{eval_type}.json")
                    process_batch_results(file_content, output_file, eval_type)
                    print(f"Results for batch ID {batch_id} saved to {output_file}")
                elif status != 'completed':
                    print(f"Batch ID {batch_id} finished with status: {status}")
        
        # Check if all batches are complete
        if all(status['complete'] for status in batch_status.values()):
            break
            
        # Check if max retries reached
        retry_count += 1
        if max_retries is not None and retry_count >= max_retries:
            print(f"Maximum retries ({max_retries}) reached. Exiting.")
            break
            
        # If no progress made in this iteration, wait before trying again
        if not any_progress:
            print(f"Waiting {poll_interval} seconds before next check...")
            time.sleep(poll_interval)
    
    # Return True if all batches are complete
    return all(status['complete'] for status in batch_status.values())

def main():
    parser = argparse.ArgumentParser(description="Retrieve results from Azure OpenAI batch jobs")
    parser.add_argument("--batch_ids", type=str, nargs="+",
                        help="Batch IDs to retrieve results for")
    parser.add_argument("--batch_ids_json", type=str,
                        help="JSON file containing batch IDs")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save results (default: outputs)")
    parser.add_argument("--eval_type", type=str, choices=["city", "country"], required=True,
                        help="Evaluation type (city or country)")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Time in seconds between polls (default: 60)")
    parser.add_argument("--max_retries", type=int,
                        help="Maximum number of retries (default: unlimited)")
    
    args = parser.parse_args()
    
    # Create client
    client = get_client()
    
    # Get batch IDs
    batch_ids = []
    if args.batch_ids:
        batch_ids = args.batch_ids
    elif args.batch_ids_json:
        with open(args.batch_ids_json, "r") as f:
            batch_ids_data = json.load(f)
            batch_ids = list(batch_ids_data.values())
    else:
        print("Error: Either --batch_ids or --batch_ids_json must be provided")
        return
    
    print(f"Found {len(batch_ids)} batch IDs to retrieve results for")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process batch jobs
    success = process_batch_jobs(
        client, 
        batch_ids, 
        args.output_dir, 
        args.eval_type,
        args.poll_interval,
        args.max_retries
    )
    
    if success:
        print("\nAll batch jobs completed successfully!")
    else:
        print("\nSome batch jobs did not complete successfully. Check logs for details.")

if __name__ == "__main__":
    main() 