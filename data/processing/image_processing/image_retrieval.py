import json
import os
import requests

def download_images_from_json(json_file_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    total_images = len(data)
    success_count = 0
    failure_count = 0
    failed_image_ids = []  # Array to store IDs of recipes with failed image download

    # Download and save images
    for idx, entry in enumerate(data):
        image_id = entry['id']
        image_url = entry['image']  # Changed to 'image'

        try:
            # Download image
            response = requests.get(image_url)
            if response.status_code == 200:
                # Save image
                with open(os.path.join(output_dir, f'{image_id}.jpg'), 'wb') as img_file:
                    img_file.write(response.content)
                success_count += 1
                print(f"Downloaded image {image_id} ({success_count}/{total_images})")
            else:
                failure_count += 1
                print(f"Failed to download image {image_id} ({failure_count} failures)")
                failed_image_ids.append(image_id)
        except Exception as e:
            failure_count += 1
            print(f"Error downloading image {image_id}: {e} ({failure_count} failures)")
            failed_image_ids.append(image_id)

    print(f"Download completed. {success_count} images downloaded, {failure_count} failures.")
    print(f"IDs of recipes with failed image download: {failed_image_ids}")

# Example usage:
json_file_path = 'filtered_recipes.json'  # Replace with the path to your JSON file
output_directory = 'images'  # Directory where images will be saved
download_images_from_json(json_file_path, output_directory)
