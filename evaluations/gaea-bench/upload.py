#!/usr/bin/env python3
"""
Script to upload files to Azure OpenAI for batch processing.
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

def upload_file(client, filename, outfolder):
    """
    Upload a file to Azure OpenAI and save the file ID.

    Args:
        client: Azure OpenAI client
        filename: Path to the file to upload
        outfolder: Output folder for file IDs
    """
    file_ids = []
    
    # Upload file with a purpose of "batch"
    with open(filename, "rb") as file_data:
        file = client.files.create(
            file=file_data,
            purpose="batch"
        )

    print(file.model_dump_json(indent=2))
    file_ids.append(file.id)
    print()

    # Save file ID
    outfile = Path(filename).name.replace('.jsonl', '.txt')
    with open(f"{outfolder}/{outfile}", "a") as f:
        f.writelines(f"{file_id}\n" for file_id in file_ids)

def main(input_dir, output_dir=None, pattern="*.jsonl"):
    """
    Main function to process files.

    Args:
        input_dir: Directory containing files to upload
        output_dir: Directory to save file IDs (defaults to input_dir/file_ids if not provided)
        pattern: File pattern to match
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(input_dir, "file_ids")
    
    # Create client
    client = get_client()
    
    # Find files to upload
    file_pattern = os.path.join(input_dir, pattern)
    files = glob(file_pattern)
    print(f"Found {len(files)} files to upload")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for filename in files:
        print(f"Processing {filename}")
        upload_file(client, filename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to Azure OpenAI for batch processing")
    parser.add_argument("--input_dir", required=True, help="Directory containing files to upload")
    parser.add_argument("--output_dir", help="Directory to save file IDs (defaults to input_dir/file_ids if not provided)")
    parser.add_argument("--pattern", default="*.jsonl", help="File pattern to match (default: *.jsonl)")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.pattern)