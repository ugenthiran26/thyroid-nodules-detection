import os
import shutil
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable Parameters
dataset_dir = 'data/p_image'  # Path to ultrasound images
mask_dir = 'data/p_mask'  # Path to mask images
output_dir = 'data/segregated'  # Path for segregated images
threshold = 28  # Threshold for classification

def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

# Create directories for sorting
benign_dir = os.path.join(output_dir, 'benign')
malignant_dir = os.path.join(output_dir, 'malignant')
create_directory(benign_dir)
create_directory(malignant_dir)

def classify_image_based_on_mask(mask_path):
    """Classify an image based on its corresponding mask."""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to load mask: {mask_path}")
            return None
        
        mean_intensity = np.mean(mask)
        label = 'malignant' if mean_intensity > threshold else 'benign'
        logging.info(f"Mask {mask_path} - Mean Intensity: {mean_intensity:.2f}, Classified as: {label}")
        return label
    except Exception as e:
        logging.error(f"Error processing mask {mask_path}: {e}")
        return None

def process_images():
    """Process and classify images based on their masks."""
    total_images = 0
    classified_images = 0
    
    for image_name in os.listdir(dataset_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            ultrasound_path = os.path.join(dataset_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)
            
            if os.path.exists(mask_path):
                label = classify_image_based_on_mask(mask_path)
                if label:
                    target_dir = benign_dir if label == 'benign' else malignant_dir
                    shutil.copy(ultrasound_path, target_dir)
                    classified_images += 1
                    logging.info(f"Moved {image_name} to {target_dir}")
                else:
                    logging.warning(f"Skipping {image_name} due to classification error.")
            else:
                logging.warning(f"No corresponding mask found for {image_name}.")
    
    logging.info(f"Processing complete. {classified_images}/{total_images} images classified and sorted.")

if __name__ == "__main__":
    process_images()
