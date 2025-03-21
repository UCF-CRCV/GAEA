#!/usr/bin/env python3
"""
Script to submit batch jobs to Azure OpenAI using file IDs.
"""
import os
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

def submit_batch(client, file_id, completion_window="24h", endpoint="/chat/completions"):
    """
    Submit a batch job using a file ID.

    Args:
        client: Azure OpenAI client
        file_id: File ID to use for the batch job
        completion_window: Time window for completion
        endpoint: API endpoint to use

    Returns:
        batch_response: Response from the batch job submission
    """
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    
    print(batch_response.model_dump_json(indent=2))
    return batch_response

def process_file_ids(client, filename, outfolder):
    """
    Process file IDs from a file and submit batch jobs.

    Args:
        client: Azure OpenAI client
        filename: File containing file IDs
        outfolder: Folder to save batch IDs

    Returns:
        batch_ids: List of batch IDs
    """
    with open(filename, "r") as f:
        file_ids = f.read().splitlines()

    batch_ids = []
    for file_id in file_ids:
        batch_response = submit_batch(client, file_id)
        batch_ids.append(batch_response.id)

    # Save batch IDs
    outfile = Path(filename).name
    with open(f"{outfolder}/{outfile}", "w") as f:
        f.writelines(f"{batch_id}\n" for batch_id in batch_ids)

    return batch_ids

def main(input_dir, output_dir=None, pattern="*.txt"):
    """
    Main function to process files.

    Args:
        input_dir: Directory containing files with file IDs
        output_dir: Directory to save batch IDs (defaults to input_dir/../batch_ids if not provided)
        pattern: File pattern to match
    """
    # Set default output directory if not provided
    if output_dir is None:
        input_path = Path(input_dir)
        output_dir = os.path.join(input_path.parent, "batch_ids")
    
    # Create client
    client = get_client()
    
    # Find files containing file IDs
    file_pattern = os.path.join(input_dir, pattern)
    files = glob(file_pattern)
    print(f"Found {len(files)} files to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for filename in files:
        print(f"Processing {filename}")
        process_file_ids(client, filename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit batch jobs to Azure OpenAI using file IDs")
    parser.add_argument("--input_dir", required=True, help="Directory containing files with file IDs")
    parser.add_argument("--output_dir", help="Directory to save batch IDs (defaults to input_dir/../batch_ids if not provided)")
    parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.pattern)