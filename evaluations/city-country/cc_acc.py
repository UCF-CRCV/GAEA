#!/usr/bin/env python3
"""
Script to calculate city and country prediction accuracy.
"""
import os
import json
import argparse
import pandas as pd

def load_scores(json_path):
    """
    Load scores from JSON file.
    
    Args:
        json_path: Path to JSON file with scores
        
    Returns:
        List of scores (integers)
    """
    scores = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            try:
                # The score is usually the last character of the output
                score = int(item['output'].strip()[-1])
                scores.append(score)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse score from item: {item['output']}. Error: {e}")
                continue
    except Exception as e:
        print(f"Error loading scores from {json_path}: {e}")
        
    return scores

def calculate_accuracy(scores):
    """
    Calculate accuracy from scores.
    
    Args:
        scores: List of scores (0 or 1)
        
    Returns:
        Accuracy as a float (0-1)
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def save_results(results, output_file):
    """
    Save results to a CSV file.
    
    Args:
        results: Dictionary of results
        output_file: Path to save results
    """
    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate city and country prediction accuracy")
    parser.add_argument("--city_file", type=str, required=True,
                        help="Path to city predictions JSON file")
    parser.add_argument("--country_file", type=str, required=True,
                        help="Path to country predictions JSON file")
    parser.add_argument("--output_file", type=str,
                        help="Path to save results as CSV (optional)")
    parser.add_argument("--model_name", type=str, default="Model",
                        help="Name of the model being evaluated")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.city_file):
        print(f"Error: City file {args.city_file} not found")
        return
    
    if not os.path.exists(args.country_file):
        print(f"Error: Country file {args.country_file} not found")
        return
    
    # Load scores
    city_scores = load_scores(args.city_file)
    country_scores = load_scores(args.country_file)
    
    # Print number of samples
    print(f"City predictions: {len(city_scores)} samples")
    print(f"Country predictions: {len(country_scores)} samples")
    
    # Calculate accuracies
    city_acc = calculate_accuracy(city_scores)
    country_acc = calculate_accuracy(country_scores)
    
    # Calculate overall accuracy
    overall_acc = (city_acc + country_acc) / 2 if city_scores and country_scores else 0.0
    
    # Print results
    print(f"Model: {args.model_name}")
    print(f"City prediction accuracy: {city_acc:.4f}")
    print(f"Country prediction accuracy: {country_acc:.4f}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    
    # Save results if output file specified
    if args.output_file:
        results = {
            "Model": args.model_name,
            "City Accuracy": city_acc,
            "Country Accuracy": country_acc,
            "Overall Accuracy": overall_acc,
            "City Samples": len(city_scores),
            "Country Samples": len(country_scores)
        }
        save_results(results, args.output_file)

if __name__ == "__main__":
    main() 