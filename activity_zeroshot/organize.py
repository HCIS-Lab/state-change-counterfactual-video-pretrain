import os
import shutil

# Set the path to the directory containing the folders
root_dir = "/nfs/wattrel/data/md0/datasets/state_aware/ucf101_images"  # Change this to your actual directory

# Loop through each item in the directory
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)

    # Ensure it's a directory and starts with "v_"
    if os.path.isdir(folder_path) and folder.startswith("v_"):
        parts = folder.split("_")  # Split by underscore (_)

        if len(parts) > 1:  # Ensure there is a second part
            category = parts[1]  # Extract the second part
            category_folder = os.path.join(root_dir, category)  # Path for new category folder

            # Create category folder if it doesn’t exist
            os.makedirs(category_folder, exist_ok=True)

            # Move the folder into the category folder
            shutil.move(folder_path, os.path.join(category_folder, folder))

print("Folders have been organized successfully!")
