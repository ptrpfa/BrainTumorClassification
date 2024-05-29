import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define the directories
source_dir = "../dataset/dataset_1"
target_dir = "../dataset/new_dataset"

# Define image augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Function to perform augmentation and save images
def augment_and_save_image(image, target_dir, file_name_prefix):
    # Reshape image to fit the expected input shape for ImageDataGenerator
    image = np.expand_dims(image, axis=0)
    
    # Generate augmented images
    augmented_images = datagen.flow(image, batch_size=1)
    
    # Save augmented images
    for i, batch in enumerate(augmented_images):
        augmented_image = batch[0].astype(np.uint8)
        file_name = f"{file_name_prefix}_aug_{i}.jpg"
        target_path = os.path.join(target_dir, file_name)
        cv2.imwrite(target_path, augmented_image)
        print(f"Augmented image saved: {target_path}")
        
        # Limiting the number of augmented images to 4 per original image
        if i >= 3:
            break

# Iterate through each subdirectory in source_dir
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # Create corresponding subdirectory in target_dir
        target_subdir = os.path.join(target_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)
        
        # Iterate through images in the subdirectory
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".jpg"):
                # Read the image
                image = cv2.imread(file_path)
                
                # Perform data augmentation and save augmented images
                augment_and_save_image(image, target_subdir, os.path.splitext(file_name)[0])
