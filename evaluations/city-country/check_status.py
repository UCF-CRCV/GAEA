#!/usr/bin/env python3
"""
Script to check the status of Azure OpenAI batch jobs for city-country evaluations.
"""
import os
import json
import time
import datetime
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

def check_batch_status(client, batch_id, poll_interval=60):
    """
    Check the status of a batch job and wait until it's complete or failed.
    
    Args:
        client: Azure OpenAI client
        batch_id: Batch ID to check
        poll_interval: Time in seconds between status checks
        
    Returns:
        batch_response: Batch response object
    """
    status = "validating"
    while status not in ("completed", "failed", "canceled", "expired"):
        time.sleep(poll_interval)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.datetime.now()} Batch ID: {batch_id}, Status: {status}")
    
    if batch_response.status == "failed":
        print("Batch job failed with errors:")
        for error in batch_response.errors.data:
            print(f"Error code {error.code}: {error.message}")
    
    return batch_response

def main():
    parser = argparse.ArgumentParser(description="Check the status of Azure OpenAI batch jobs")
    parser.add_argument("--batch_ids", type=str, nargs="+",
                        help="Batch IDs to check status for")
    parser.add_argument("--batch_ids_json", type=str,
                        help="JSON file containing batch IDs")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Time in seconds between status checks (default: 60)")
    
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
    
    print(f"Found {len(batch_ids)} batch IDs to check")
    
    # Check status for each batch ID
    for batch_id in batch_ids:
        print(f"\nChecking status for batch ID: {batch_id}")
        batch_response = check_batch_status(client, batch_id, args.poll_interval)
        print(f"Final status for batch ID {batch_id}: {batch_response.status}")

if __name__ == "__main__":
    main() 