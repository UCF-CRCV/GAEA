import os
import json
import time
import copy
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm 
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


warnings.filterwarnings("ignore")

# Create a custom dataset class to load the data
class MultimodalDataset(Dataset):
    def __init__(self, img_root, dataframe, transform=None):
        self.img_root = img_root
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.dataframe.iloc[idx]['image']
        question = self.dataframe.iloc[idx]['conversations'][0]["value"]
        answer = self.dataframe.iloc[idx]['conversations'][1]["value"]
        question_type = self.dataframe.iloc[idx]['question_type']
        unique_id = self.dataframe.iloc[idx]['unique_id']
        
        try:
            ima_path = os.path.join(self.img_root, image)
        except Exception as e:
            print(f"Error in loading image: {e}")
            print(f"Image : {ima_path}")

        
        return {
            'image_path': ima_path,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'unique_id': unique_id
        }

# Load data from json file
def load_data(file, select_few):
    df = pd.DataFrame(json.load(open(file)))
    if select_few:
        return df[df["question_type"].apply(lambda x: x in select_few)]
    return df

def default_converter(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def append_json_line(file_path, data):
    """
    Append a single JSON object as a new line in the JSONL file.
    
    Parameters:
        file_path (str): Path to the JSONL file.
        data (dict): The JSON object to append.
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data, default=default_converter)
        f.write(json_line + '\n')
        f.flush()          # Ensure data is written to the buffer
        os.fsync(f.fileno())  # Force write to disk

def read_json_lines(file_path):
    """
    Read JSON objects from a JSONL file, skipping any malformed lines.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: A list of successfully parsed JSON objects.
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                json_obj = json.loads(line)
                results.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                print(f"Skipping malformed line: {line}")
    return results


# Custom collate function to handle varying image sizes and batch data
def collate_fn(batch):
    images = [item['image_path'] for item in batch]  # Image paths
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    question_types = [item['question_type'] for item in batch]
    unique_ids = [item['unique_id'] for item in batch]

    return {
        'images': images,
        'questions': questions,
        'answers': answers,
        'question_types': question_types,
        'unique_ids': unique_ids

    }

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dynamic_prompt_json(prompt_type):
    if prompt_type == 'LVQA':
        return 'Drawing upon your expertise in geography and tourism, examine the image and provide a comprehensive description of the community or lifestyle depicted. Include insights about cultural practices, geographic features, terrain, local flora and fauna, infrastructure, and any natural or man-made elements that characterize the location. Consider how these factors influence the lifestyle and community in the area. Question: '
    elif prompt_type == 'SVQA':
        return 'Provide a short answer on notable landmarks, museums, parks, restaurants, or activities that visitors might enjoy in the area. Highlight amenities and services that enhance the tourism experience at this location. Question: '
    elif prompt_type == 'MCQ':
        return 'Use your comprehensive knowledge of geography, landmarks, and tourism to analyze the image and determine the correct answer from the options provided. Note, your final answer should be a choice of either A, B, C, or D, including both the letter and the complete text of the option exactly as presented.  Question: '
    elif prompt_type == 'TF':
        return "Use your comprehensive knowledge of notable landmarks, museums, parks, restaurants, and related attractions to evaluate the following statement. Provide your final answer as either 'True' or 'False'. Question: "

# Evaluate the model and collect predictions, ground truth, and scores
def evaluate(model, dataloader, device, results, min_pixels, max_pixels, tmp_save_path):
    model.eval()

    already_processed = set([res["filename"] for res in results])

    dynamic_prompt = dynamic_prompt_json

    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_paths = batch['images']
            questions = batch['questions']
            ground_truth = batch['answers']
            question_types = batch['question_types']
            unique_ids = batch['unique_ids']

            for img_path, question, answer, question_type, unique_id in zip(image_paths, questions, ground_truth, question_types, unique_ids):
                # Prepare input with image and question
                    filename = Path(img_path).name
                    if filename in already_processed:
                        continue                 

                    prompt = dynamic_prompt(question_type)
                    

                    new_question = question.replace('<image>\n', '')
                    assert not new_question.startswith('<image>\n') and '<image>\n' not in new_question, "This tag is not valid."

                    user_prompt = prompt + f"{new_question}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image", 
                                    "image": img_path,
                                    'min_pixel':min_pixels,
                                    'max_pixel':max_pixels
                                },
                                {"type": "text", "text": user_prompt},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(device)

                    # Inference: Generation of the output
                    generated_ids = model.generate(**inputs, max_new_tokens=1200, temperature=0.2)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    predicted_answer = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    output_dict = {
                        "filename": filename,
                        "predicted_answer": predicted_answer,
                        "ground_truth": answer,
                        "question": question,
                        "question_type": question_type,
                        "unique_id": unique_id
                    }
                    results.append(output_dict)

                    append_json_line(tmp_save_path, output_dict)
    return results



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the json containing the MP16 evaluation set", default="")
    parser.add_argument("--img_root", type=str, help="Path to the MP16 images", default="/home/c3-0/datasets/MP-16/resources/images/mp16/")
    parser.add_argument("--pretrained", type=str, default="", help="Path to the pretrained model weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Save your results to your directory")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    file_path = args.file_path
    img_root = args.img_root
    save_dir = args.save_dir


    os.makedirs(save_dir, exist_ok=True)

    pretrained = args.pretrained
    device = args.device
    model_base= args.model_base
    attn_implementation = args.attn_implementation

    pretrained = model_base if pretrained == "" else pretrained

    min_pixels = 224*224
    max_pixels = 2048*2048

    print(f'Inference on model: {pretrained}')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype='auto', low_cpu_mem_usage=True, attn_implementation='flash_attention_2', device_map="auto")
    processor = AutoProcessor.from_pretrained(pretrained, padding_side="right", min_pixels=min_pixels, max_pixels=max_pixels)
    model.eval()

    time1 = time.time()

    df = load_data(file_path, None)    
    # Initialize dataset and dataloader
    dataset = MultimodalDataset(img_root, df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    tmp_save_path = os.path.join(save_dir, "GAEA-tmp_predictions.jsonl")

    if os.path.exists(tmp_save_path):
        results = read_json_lines(tmp_save_path)
    else:
        results = []

    # Perform evaluation
    results = evaluate(model, dataloader, device, results, min_pixels, max_pixels, tmp_save_path)
    save_path = os.path.join(save_dir, "GAEA-predictions.json")
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    print("Results saved successfully.")
    print(f"Time taken: {time.time() - time1} seconds")