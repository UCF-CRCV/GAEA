#!/usr/bin/env python3
"""
Script to retrieve results from Azure OpenAI batch jobs.
Continues polling until all jobs are complete.
"""
import json
import os
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
    print(f"{datetime.datetime.now()} Batch Id: {batch_id}, Status: {status}")

    output_file_id = batch_response.output_file_id
    if not output_file_id:
        output_file_id = batch_response.error_file_id

    file_content = None
    if output_file_id:
        file_response = client.files.content(output_file_id)
        print(f'Batch id: {batch_id} for input file: {batch_response.input_file_id}')
        file_content = file_response.text
    
    return batch_response, file_content, status

def process_batch_ids(client, filename, outfolder, poll_interval=60, max_retries=None):
    """
    Process batch IDs from a file and retrieve results.
    Continues polling until all jobs are complete.

    Args:
        client: Azure OpenAI client
        filename: File containing batch IDs
        outfolder: Folder to save results
        poll_interval: Time in seconds between polls
        max_retries: Maximum number of retries (None for unlimited)

    Returns:
        bool: True if all jobs completed successfully, False otherwise
    """
    with open(filename, "r") as f:
        batch_ids = f.read().splitlines()

    outfile = Path(filename).name.replace('.txt', '.jsonl')
    output_path = f"{outfolder}/{outfile}"
    
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
                    # If this is the first content to be written to this file
                    if not os.path.exists(output_path):
                        with open(output_path, "w") as f:
                            f.write(file_content)
                    else:
                        # Append to existing file
                        with open(output_path, "a") as f:
                            f.write(file_content)
                    print(f"Results for batch ID {batch_id} saved to {output_path}")
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

def main(input_dir, output_dir=None, pattern="*.txt", poll_interval=60, max_retries=None):
    """
    Main function to process files.

    Args:
        input_dir: Directory containing files with batch IDs
        output_dir: Directory to save results (defaults to input_dir/../jsonl_responses if not provided)
        pattern: File pattern to match
        poll_interval: Time in seconds between polls
        max_retries: Maximum number of retries (None for unlimited)
    """
    # Set default output directory if not provided
    if output_dir is None:
        input_path = Path(input_dir)
        output_dir = os.path.join(input_path.parent, "jsonl_responses")
    
    # Create client
    client = get_client()
    
    # Find files containing batch IDs
    file_pattern = os.path.join(input_dir, pattern)
    files = glob(file_pattern)
    print(f"Found {len(files)} files to process: {files}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    all_success = True
    for filename in files:
        print(f"\nProcessing {filename}")
        success = process_batch_ids(client, filename, output_dir, poll_interval, max_retries)
        all_success = all_success and success
    
    if all_success:
        print("\nAll batch jobs completed successfully!")
    else:
        print("\nSome batch jobs did not complete successfully. Check logs for details.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve results from Azure OpenAI batch jobs")
    parser.add_argument("--input_dir", required=True, help="Directory containing files with batch IDs")
    parser.add_argument("--output_dir", help="Directory to save results (defaults to input_dir/../jsonl_responses if not provided)")
    parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    parser.add_argument("--poll_interval", type=int, default=60, help="Time in seconds between polls (default: 60)")
    parser.add_argument("--max_retries", type=int, help="Maximum number of retries (default: unlimited)")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.pattern, args.poll_interval, args.max_retries)
    