#!/usr/bin/env python3
"""
This script evaluates the accuracy of location predictions by geocoding predicted locations
and calculating distances between predicted and actual coordinates. It uses the Nominatim
geocoding service to convert location names to coordinates.

Usage:
    python dist_acc.py --pred_path <path_to_predictions> --outfolder <output_folder> --outfile <output_file>
"""

import csv
import json
import os
from geopy.distance import distance
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
import numpy as np
import argparse
import uuid


def initialize_geocoder():
    """
    Initialize and return a Nominatim geocoder with a unique user agent.
    
    Returns:
        Nominatim: Initialized geocoder object
    """
    # Create a unique user agent for Nominatim to avoid rate limiting
    random_user = f"my_user_{str(uuid.uuid4())}"
    return Nominatim(user_agent=random_user)


def load_existing_predictions(results_path):
    """
    Load existing geocoded predictions if available.
    
    Args:
        results_path: Path to the JSON file with existing predictions
        
    Returns:
        tuple: (predictions dictionary, set of processed filenames)
    """
    if os.path.exists(results_path):
        preds = json.load(open(results_path))
        pred_set = set(preds.keys())
        return preds, pred_set
    else:
        return {}, set()


def save_predictions(preds, results_path):
    """
    Save predictions to a JSON file.
    
    Args:
        preds: Dictionary of predictions
        results_path: Path to save predictions
    """
    json.dump(preds, open(results_path, "w"), indent=4)


def geocode_predictions(results, geocoder, preds, pred_set, results_path, save_interval=100):
    """
    Geocode location predictions and store coordinates.
    
    Args:
        results: List of prediction results to process
        geocoder: Nominatim geocoder object
        preds: Dictionary to store geocoded predictions
        pred_set: Set of already processed filenames
        results_path: Path to save results
        save_interval: Interval to save intermediate results
        
    Returns:
        tuple: (updated predictions, dictionary of failed geocoding attempts)
    """
    faulty = {}
    print("Geocoding predictions...")
    
    for idx, item in enumerate(tqdm(results, desc='Calculating distances')):
        # Skip already processed predictions
        if item['filename'] in pred_set:
            continue
        
        # Get the predicted location and try to geocode it
        prediction = item['predicted_answer'].strip()
        try:
            gps = geocoder.geocode(prediction, timeout=60)
            if gps:
                preds[item['filename']] = [gps.latitude, gps.longitude]
            else:
                faulty[item['filename']] = prediction
        except Exception as e:
            faulty[item['filename']] = prediction
            
        # Save intermediate results at specified intervals
        if idx % save_interval == 0 and idx > 0:
            save_predictions(preds, results_path)
    
    return preds, faulty


def process_predictions(pred_path, outfolder, outfile):
    """
    Main function to process and geocode predictions.
    
    Args:
        pred_path: Path to the JSON file with predictions
        outfolder: Folder to save results
        outfile: Output file name
        
    Returns:
        tuple: (predictions dictionary, dictionary of failed geocoding attempts)
    """
    # Initialize geocoder
    geocoder = initialize_geocoder()
    
    # Create output directory if it doesn't exist
    save_dir = os.path.join("saved_gecoding", outfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up results path
    results_path = os.path.join(save_dir, outfile)
    print(f"Results will be saved to: {results_path}")
    
    # Load the model predictions
    with open(pred_path, 'r') as f:
        results = json.load(f)
    
    # Load existing predictions if available
    preds, pred_set = load_existing_predictions(results_path)
    
    # Process predictions
    preds, faulty = geocode_predictions(results, geocoder, preds, pred_set, results_path)
    
    # Save final results
    save_predictions(preds, results_path)
    
    return preds, faulty


def main():
    """Main entry point to run the script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Geocode location predictions and calculate distances")
    parser.add_argument("--pred_path", required=True, type=str, help='Path to the JSON file containing model predictions')
    parser.add_argument("--outfolder", required=True, type=str, help='Name of the folder to save results')
    parser.add_argument("--outfile", required=True, type=str, help='Name of the output JSON file')
    args = parser.parse_args()
    
    print(f"Processing predictions from: {args.pred_path}")
    
    # Process predictions
    preds, faulty = process_predictions(args.pred_path, args.outfolder, args.outfile)
    
    # Print failed geocoding attempts
    print('Faulty predictions (could not be geocoded):')
    for filename, prediction in faulty.items():
        print(f"{filename}: {prediction}")


if __name__ == "__main__":
    main()

