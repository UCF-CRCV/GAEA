#!/usr/bin/env python3
"""
Script to submit batch jobs to Azure OpenAI for city-country evaluations.
"""
import os
import json
import argparse
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

def submit_batch(client, file_id, completion_window="24h", endpoint="/chat/completions"):
    """
    Submit a batch job using a file ID.
    
    Args:
        client: Azure OpenAI client
        file_id: File ID to use for the batch job
        completion_window: Time window for completion
        endpoint: API endpoint to use
        
    Returns:
        batch_id: ID of the created batch
    """
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    
    print(f"Created batch job with ID: {batch_response.id} for file ID: {file_id}")
    print(batch_response.model_dump_json(indent=2))
    
    return batch_response.id

def main():
    parser = argparse.ArgumentParser(description="Submit batch jobs to Azure OpenAI")
    parser.add_argument("--file_ids", type=str, nargs="+",
                        help="File IDs to submit batch jobs for")
    parser.add_argument("--file_ids_json", type=str,
                        help="JSON file containing file IDs")
    parser.add_argument("--output_file", type=str, default="batch_ids.json",
                        help="Output file to save batch IDs")
    parser.add_argument("--completion_window", type=str, default="24h",
                        help="Completion window for batch jobs (default: 24h)")
    
    args = parser.parse_args()
    
    # Create client
    client = get_client()
    
    # Get file IDs
    file_ids = []
    if args.file_ids:
        file_ids = args.file_ids
    elif args.file_ids_json:
        with open(args.file_ids_json, "r") as f:
            file_ids_data = json.load(f)
            file_ids = list(file_ids_data.values())
    else:
        print("Error: Either --file_ids or --file_ids_json must be provided")
        return
    
    print(f"Found {len(file_ids)} file IDs to process")
    
    # Submit batch jobs
    batch_ids = {}
    for file_id in file_ids:
        batch_id = submit_batch(client, file_id, args.completion_window)
        batch_ids[file_id] = batch_id
    
    # Save batch IDs to output file
    with open(args.output_file, "w") as f:
        json.dump(batch_ids, f, indent=2)
    
    print(f"Saved batch IDs to {args.output_file}")

if __name__ == "__main__":
    main() 