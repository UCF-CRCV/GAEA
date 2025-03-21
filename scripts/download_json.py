import requests
import argparse
import os

def download_file(base_url, filename, save_dir):
    """Download a file from the dataset and save it locally."""
    file_url = f"{base_url.rstrip('/')}/{filename}"
    save_path = os.path.join(save_dir, filename)

    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an error if the download fails

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the file
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded: {filename} -> {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download specific files from a dataset.")
    parser.add_argument("--file_name",type=str, help="List of filenames to download (e.g., GAEA-Bench.json GAEA-Train.json)")
    parser.add_argument("--base-url", required=True, help="Base URL of the dataset")
    parser.add_argument("--save-dir", default=".", help="Directory to save the downloaded files (default: current directory)")
    
    args = parser.parse_args()


    download_file(args.base_url, args.file_name, args.save_dir)

if __name__ == "__main__":
    main()
