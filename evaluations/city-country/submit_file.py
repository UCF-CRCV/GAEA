#!/usr/bin/env python3
"""
Script to upload files to Azure OpenAI for batch processing city-country evaluations.
"""
import os
import argparse
import json
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

def upload_file(client, file_path):
    """
    Upload a file to Azure OpenAI for batch processing.
    
    Args:
        client: Azure OpenAI client
        file_path: Path to the file to upload
        
    Returns:
        file_id: ID of the uploaded file
    """
    with open(file_path, "rb") as file_data:
        file = client.files.create(
            file=file_data,
            purpose="batch"
        )
    
    print(f"Uploaded {file_path} with ID: {file.id}")
    return file.id

def main():
    parser = argparse.ArgumentParser(description="Upload files to Azure OpenAI for batch processing")
    parser.add_argument("--input_dir", type=str, default="jsonls", 
                        help="Directory containing JSONL files to upload")
    parser.add_argument("--file_list", type=str, nargs="+",
                        help="Specific files to upload (if not provided, all JSONL files in input_dir will be used)")
    parser.add_argument("--output_file", type=str, default="file_ids.json",
                        help="Output file to save file IDs")
    
    args = parser.parse_args()
    
    # Create client
    client = get_client()
    
    # Get files to upload
    if args.file_list:
        files = args.file_list
    else:
        files = glob(os.path.join(args.input_dir, "*.jsonl"))
    
    print(f"Found {len(files)} files to upload")
    
    # Upload files and save file IDs
    file_ids = {}
    for file_path in files:
        file_id = upload_file(client, file_path)
        file_ids[os.path.basename(file_path)] = file_id
    
    # Save file IDs to output file
    with open(args.output_file, "w") as f:
        json.dump(file_ids, f, indent=2)
    
    print(f"Saved file IDs to {args.output_file}")

if __name__ == "__main__":
    main() 