import os
import base64
import threading
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import time
from openai import OpenAI


# Global thread lock
file_lock = threading.Lock()


def openai_api(image_path, system_prompt, model_name):
    """Call OpenAI API to generate image caption using standard OpenAI client"""

    api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Convert image to base64
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
        ],
    )

    return response.choices[0].message.content


def vllm_api(image_path, system_prompt, model_name="Qwen3-VL-30B-A3B"):
    """Call vLLM-compatible API to generate image caption"""

    api_key = os.environ.get("VLLM_API_KEY", "YOUR_API_KEY_HERE")
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8080/v1")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Convert image to base64
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
        ],
    )

    return response.choices[0].message.content


def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption(image_path, model_name, max_retries=3):
    """Generate image caption using LVLM"""

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate image format
    try:
        Image.open(image_path)
    except IOError:
        raise ValueError(f"Unsupported image format: {image_path}")

    # System prompt definition
    system_prompt = """
    You are an image caption writer. Your task is to describe the character in the image in a single, clear, and detailed paragraph.

    ## Instructions:

    **DO:**

    * Describe the character's gender and artistic style briefly.
    * Describe the character from head to toe, focusing on the layers of clothing and materials.
    * Break down the description into sections based on body parts (e.g., head, left arm, right arm, torso, left leg, right leg).
    * Pay special attention to the head, providing a detailed description.
    * For symmetrical body parts, describe them individually (e.g., describe each wrist separately if both have accessories).
    * Be specific about colors, materials (e.g., metal, fabric, leather), and how the clothing layers are arranged.

    **DON'T:**

    * Do not use overly technical or obscure words.
    * Avoid long, complicated sentences.
    * Do not invent details that are not visible in the image.

    ## Perspective Note:

    The left side of the image is the character's right side, and vice-versa.

    **Now, write a caption for the image I've provided.**
    """

    for attempt in range(max_retries):
        try:
            if "gpt" in model_name:
                # Call OpenAI API
                content = openai_api(image_path, system_prompt, model_name)
            else:
                # Call vLLM-compatible API
                content = vllm_api(image_path, system_prompt, model_name)

            return content

        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                delay = 10
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print("All retry attempts failed")
                return {"reason": f"API call failed: {str(e)}", "result": "no"}


def load_checkpoint(output_file):
    """Load list of already processed images"""
    processed_files = set()
    if os.path.exists(output_file):
        try:
            with file_lock:
                with open(output_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and 'img_name' in reader.fieldnames:
                        processed_files = {row['img_name'] for row in reader if row['img_name']}
        except Exception:
            backup_file = output_file + '.backup'
            with file_lock:
                if os.path.exists(output_file):
                    os.rename(output_file, backup_file)
                    print(f"Original file format error, backed up as {backup_file}")
    return processed_files


def append_result_thread_safe(result, output_file):
    """Thread-safe append result to CSV file with image_id field"""
    with file_lock:
        file_exists = os.path.exists(output_file)

        with open(output_file, 'a', encoding='utf-8-sig', newline='') as f:
            # Field names including image_id
            writer = csv.DictWriter(f, fieldnames=['image_id', 'img_name', 'caption', 'error'])

            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write data row
            writer.writerow(result)


def process_single_image(image_file, folder_path, output_file, image_id, model_name):
    """Process a single image with image_id support"""
    image_path = os.path.join(folder_path, image_file)
    image_path = image_path.replace("\\", "/")

    try:
        caption = generate_caption(image_path, model_name)

        # Create result dict with image_id
        result_dict = {
            "image_id": image_id,
            "img_name": image_file,
            "caption": caption,
            "error": ""
        }

        # Thread-safe save result
        append_result_thread_safe(result_dict, output_file)

        return {"status": "success", "result": result_dict}

    except Exception as e:
        # Record failed file with image_id
        failed_dict = {
            "image_id": image_id,
            "img_name": image_file,
            "caption": "",
            "error": str(e)
        }
        append_result_thread_safe(failed_dict, output_file)
        return {"status": "error", "result": failed_dict, "error": str(e)}


def process_images_in_folder_multithread(folder_path, output_file="captions.csv",
                                         resume=True, max_workers=4, model_name="Qwen3-VL-30B-A3B"):
    """Multi-threaded batch processing of images in a folder with sorting support"""

    supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    # Get all image files including subdirectories, preserving original order
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in supported_formats):
                # Get relative path from folder_path
                rel_path = os.path.relpath(os.path.join(root, f), folder_path)
                image_files.append(rel_path)

    if not image_files:
        print("No supported image formats found in the folder")
        return []

    # Create indexed list for image files [(id, filename), ...]
    indexed_files = [(i, f) for i, f in enumerate(image_files)]

    # Load processed files
    processed_files = set()
    if resume:
        processed_files = load_checkpoint(output_file)
        if processed_files:
            print(f"Found {len(processed_files)} already processed images, continuing with remaining")

    # Filter unprocessed files, preserving index
    remaining_files = [(idx, f) for idx, f in indexed_files if f not in processed_files]

    if not remaining_files:
        print("All images have been processed")
        results = []
        if os.path.exists(output_file):
            with file_lock:
                with open(output_file, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    results = list(reader)
        return results

    print(f"Total {len(image_files)} images, {len(remaining_files)} remaining to process")

    # Use thread pool to process images
    success_count = 0
    failed_count = 0
    failed_files = []

    # Store results for sorting by ID
    results_with_id = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with image_id and model_name
        future_to_data = {
            executor.submit(process_single_image, image_file, folder_path, output_file, image_id, model_name): (image_id, image_file)
            for image_id, image_file in remaining_files
        }

        # Use tqdm to show progress
        with tqdm(total=len(remaining_files), desc="Processing", unit="img") as pbar:
            for future in as_completed(future_to_data):
                image_id, image_file = future_to_data[future]
                try:
                    result = future.result()
                    if result["status"] == "success":
                        success_count += 1
                        results_with_id.append(result["result"])
                    else:
                        failed_count += 1
                        failed_files.append((image_id, image_file))
                except Exception as e:
                    failed_count += 1
                    failed_files.append((image_id, image_file))
                    print(f"Exception while processing {image_file} (ID: {image_id}): {e}")

                pbar.update(1)

    # Print statistics
    if failed_files:
        print(f"Done! {success_count} succeeded, {failed_count} failed")
        print(f"Failed files: {failed_files}")
    else:
        print(f"Done! All {success_count} images processed successfully")

    # Return results sorted by ID
    if os.path.exists(output_file):
        with file_lock:
            with open(output_file, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                results = list(reader)
                # Sort by image_id
                results.sort(key=lambda x: int(x['image_id']))

            # Write sorted results back to file
            with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            return results

    return []


# Keep single-threaded function for backward compatibility
def process_images_in_folder(folder_path, output_file="captions.csv", resume=True, model_name="Qwen3-VL-30B-A3B"):
    """Batch process images in a folder (single-threaded, backward compatible)"""
    return process_images_in_folder_multithread(folder_path, output_file, resume, max_workers=1, model_name=model_name)


def get_single_image_caption(image_path, model_name="Qwen3-VL-30B-A3B"):
    """Get caption for a single image and return as dict"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    filename = os.path.basename(image_path)

    try:
        caption = generate_caption(image_path, model_name)
        result = {
            "img_name": filename,
            "caption": caption,
            "error": ""
        }
        return result
    except Exception as e:
        result = {
            "img_name": filename,
            "caption": "",
            "error": str(e)
        }
        return result


def clear_checkpoint(output_file="captions.csv"):
    """Clear checkpoint file to restart processing"""
    with file_lock:
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Checkpoint file cleared: {output_file}")
        else:
            print("Checkpoint file does not exist")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image caption generation with CLI support"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing images to process",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="captions.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of concurrent threads for processing",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-VL-30B-A3B",
        help='Model name, e.g. "gpt-xxx", "Qwen3-VL-30B-A3B"',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[Config] folder_path   = {args.folder_path}")
    print(f"[Config] output_file   = {args.output_file}")
    print(f"[Config] model_name    = {args.model_name}")
    print(f"[Config] max_workers   = {args.max_workers}")

    process_images_in_folder_multithread(
        folder_path=args.folder_path,
        output_file=args.output_file,
        max_workers=args.max_workers,
        model_name=args.model_name,
    )