import os
import argparse
from io import BytesIO
from PIL import Image
from datasets import load_dataset

# Define dataset-to-folder mapping
DATASET_MAPPING = {
    "MP16": "MP-16",
    "GLDv2": "GLDv2",
    "CityGuessr": "CityGuessr"
}

def save_image(file_data, image_path):
    """Save the image from a PIL object or byte data to the specified path."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    if isinstance(file_data, Image.Image):
        image = file_data
    else:
        image = Image.open(BytesIO(file_data))

    image.save(image_path)

def process_training_data(dataset, root_dir):
    """Process training data by preserving folder structure."""
    for entry in dataset:
        dataset_type = entry["dataset"]
        image_path = entry["image"]  # Example: '00/22/23423.jpg'
        file_bytes = entry["file_name"]  # Compressed image data

        # Validate dataset type
        if dataset_type not in DATASET_MAPPING:
            continue
        
        target_path = os.path.join(root_dir, DATASET_MAPPING[dataset_type], image_path)
        save_image(file_bytes, target_path)

def process_testing_data(dataset, root_dir):
    """Process testing data by saving images directly in the root directory."""
    for entry in dataset:
        image_path = os.path.join(root_dir, entry["image"])
        file_bytes = entry["file_name"]  # Compressed image data
        save_image(file_bytes, image_path)

def main():
    parser = argparse.ArgumentParser(description="Download and organize Hugging Face dataset.")
    parser.add_argument('--root_dir', type=str, help="Root directory to save the dataset")
    parser.add_argument("--dataset_name", type=str, help="Name of the Hugging Face dataset")
    parser.add_argument("--split", type=str, help="Dataset split to process")
    args = parser.parse_args()

    # Define root directory
    root_dir = args.root_dir
    os.makedirs(root_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)

    # Process data
    if args.split == "train":
        process_training_data(dataset, root_dir)
    else:
        process_testing_data(dataset, root_dir)

    print(f"{args.split.capitalize()} data processing complete. Files saved under {root_dir}.")

if __name__ == "__main__":
    main()
