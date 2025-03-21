#!/usr/bin/env python3
"""
Script to evaluate model predictions using GPT-4 as a judge.
"""
import os
import json
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path
from multiprocessing.pool import Pool


def prompt_type(question_type, question, ground_truth, prediction):
    """
    Generate an evaluation prompt based on the question type.
    
    Args:
        question_type: Type of question (SVQA, LVQA, MCQ, TF)
        question: The question text
        ground_truth: The ground truth answer
        prediction: The model's predicted answer
        
    Returns:
        str: Formatted prompt for evaluation
    """
    if question_type == "SVQA":
        return (
            '''Evaluate the following predicted answer by comparing it to the provided ground truth. Focus on the criteria below:\n''' \
            '''1. Location Prediction Correct: How accurately does the predicted answer identify the specific country, city, state mentioned in the ground truth. Accept responses that geographically contain the ground-truth, e.g. U.K. for Scotland or England. Accept responses that are synonyms or near synonyms, e.g. Brooklyn is part of New York City. Give partial credit if the guess is less specific but close to correct, e.g. Western Australia for Perth.\n''' \
            '''2. Specificity and Relevance: Does the predicted answer provide specific information that is directly relevant to the question and closely aligns with the ground truth? Vague or overly general descriptions that lack alignment should result in a lower score.\n''' \
            '''Scoring Instructions:\n''' \
            '''Additional Instructions for Consistency:\n''' \
            '''- Provide the score out of 10 for each above criteria in the order listed.\n''' \
            '''- Vague but verbose descriptions should not count positively and should lower the overall score.\n''' \
            '''Only provide the numeric score, without any additional commentary.\n''' \
            '''- Provide the final output as a Python dictionary with the structure only dont add a anything extra, because your output will be used in code pipeline. So single change in your output will crash whole system. \n''' \
            
            '''# Example output : {'Location_Prediction': 10.0, 'Specificity_and_Relevance': 9.5}''' \
            '''---\n''' \
            f"""**Question**: "{question}"\n""" \
            f"""**Ground Truth**: "{ground_truth}"\n""" \
            f"""**Model Prediction**: "{prediction}" """
        )
    elif question_type == "LVQA":
        return '''Evaluate the following predicted answer by comparing it to the provided ground truth. Focus on the criteria below:\n''' \
                        '''1. **Location Relevance (Highest Weight)**: How accurately does the predicted answer identify and describe the specific location mentioned in the ground truth? This includes correct geographical identification and location-specific details.\n''' \
                        '''2. **Cultural Aspect Matching**: How well does the predicted answer capture and reflect the cultural aspects present in the ground truth? This includes local customs, architectural styles, landmarks, historical context, and other culturally significant elements.\n''' \
                        '''3. **Consistency and Quality of Reasoning**: Is the predicted answer logically consistent and does it demonstrate sound reasoning based on the information provided? Does it correctly interpret and infer details as presented in the ground truth?\n''' \
                        '''4. **Specificity and Relevance**: Does the predicted answer provide specific information that is directly relevant to the question and closely aligns with the ground truth? Vague or overly general descriptions that lack alignment should result in a lower score.\n''' \
                        '''5. **Fluency and Clarity**: Is the language in the predicted answer fluent, clear, and well-articulated?\n''' \
                        '''**Scoring Instructions:**\n\n''' \
                        '''- Provide a single overall score out of **10**, weighing the criteria in the order listed, with **Location Relevance** and **Cultural Aspect Matching** receiving the most weight.\n''' \
                        '''- Be rigorous in your evaluation and **do not be lenient**. Penalize the predicted answer for any vagueness, lack of specificity, or irrelevance.\n''' \
                        '''- **Vague but verbose descriptions should not count positively** and should lower the overall score.\n\n''' \
                        '''**Only provide the numeric score**, without any additional commentary.\n''' \
                        '''---\n''' \
                        f'''**Question**: {question}\n''' \
                        f'''**Ground Truth**: {ground_truth}\n''' \
                        f'''**Model Prediction**: {prediction}'''
    elif question_type in ['MCQ', 'TF']:
        return (
            "Evaluate the following answer based on Accuracy:\n\n"
            f"Question: {question}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Model Prediction: {prediction}\n\n"
            "Match the meaning of the ground truth with the model prediction and if it matches give a 10. Otherwise 0.\n"
            "Strictly return only the numeric score, without any additional commentary."
        )

def create_evaluation_batch(data, outfile, question_type):
    """
    Create an evaluation batch entry for GPT-4.
    
    Args:
        data: Entry data containing question, answer, etc.
        outfile: Output file name
        question_type: Type of question
        
    Returns:
        dict: Batch entry for GPT evaluation
    """
    question = data['question'].replace('<image>\n', '')
    
    if 'GAEA' in data.get('model_name', '') or outfile.startswith('GAEA'):
        prediction = data['predicted_answer'][0].strip()
    else:
        prediction = data['predicted_answer'].strip()
        
    ground_truth = data['ground_truth']
    prompt = prompt_type(question_type, question, ground_truth, prediction)
    
    if question_type == 'SVQA':
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "EvaluationScores",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "Location_Prediction": {"type": "number"},
                        "Specificity_and_Relevance": {"type": "number"}
                    },
                    "required": [
                        "Location_Prediction",
                        "Specificity_and_Relevance"
                    ],
                    "additionalProperties": False
                }
            }
        }
        
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": prompt
            }
        ]
        
        return {
            "custom_id": f'{outfile.replace("-predictions.jsonl","")}_{data["unique_id"]}', 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o-mini", 
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1000
            },
            "response_format": response_format
        }
    else:
        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": prompt
            }
        ]
        
        return {
            "custom_id": f'{outfile.replace("-predictions.jsonl","")}_{data["unique_id"]}', 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": "gpt-4o-mini", 
                "messages": PROMPT_MESSAGES,
                "max_tokens": 1000
            },
        }

def process_json_file(json_path, outfolder):
    """
    Process a single JSON file with predictions.
    
    Args:
        json_path: Path to the JSON file with predictions
        outfolder: Folder to save output JSONL files
    """
    print(f"Processing {json_path}")
    qa_json = json.load(open(json_path))
    
    batch = []
    outfile = Path(json_path).name.replace('.json', '.jsonl')
    
    for idx, data in enumerate(tqdm(qa_json)):
        question_type = data['question_type']
        batch_entry = create_evaluation_batch(data, outfile, question_type)
        batch.append(batch_entry)
    
    output_path = os.path.join(outfolder, outfile)
    with open(output_path, 'w') as f:
        for entry in batch:
            f.write(json.dumps(entry))
            f.write('\n')
    
    print(f"Created evaluation batch: {output_path}")

def main(json_path=None, json_folder=None, outfolder=None):
    """
    Main function to process files.
    
    Args:
        json_path: Path to a single JSON file with predictions
        json_folder: Path to a folder containing JSON files with predictions
        outfolder: Folder to save output JSONL files
    """
    # Create output directory if it doesn't exist
    os.makedirs(outfolder, exist_ok=True)
    
    if json_path:
        process_json_file(json_path, outfolder)
    elif json_folder:
        pred_files = glob(os.path.join(json_folder, '*.json'), recursive=True)
        print(f"Found {len(pred_files)} prediction files")
        
        for pred_file in pred_files:
            process_json_file(pred_file, outfolder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare evaluation batches for GPT-4 judge")
    parser.add_argument("--json_path", help="Location of a single prediction file")
    parser.add_argument("--json_folder", help="Folder where prediction files are located")
    parser.add_argument("--outfolder", required=True, help="Folder which all files will be saved")
    
    args = parser.parse_args()
    
    if not (args.json_path or args.json_folder):
        parser.error("At least one of --json_path or --json_folder must be provided")
    
    main(args.json_path, args.json_folder, args.outfolder)
