import os
import shutil
import pandas as pd
from PIL import Image
from torchvision import transforms

# Load the dataset
full_dataset = pd.read_csv('../faces-age/train.csv')

# Define the augmentation transformations
augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # You can add more transformations here if needed
])

# Create the directory for augmented data if it does not exist
augmented_root_dir = '../faces-age/data_augmented'
os.makedirs(augmented_root_dir, exist_ok=True)

# Copy original images to the augmented data directory
for filename in full_dataset['filename']:
    original_image_path = os.path.join('../faces-age', filename)
    destination_image_path = os.path.join(augmented_root_dir, filename)
    shutil.copy2(original_image_path, destination_image_path)

# Calculate the target count based on the distribution
age_counts = full_dataset['age'].value_counts()
target = age_counts.median()  # Or another strategy to determine the target count

# Store new rows for augmented data
augmented_rows = []

# Augment the data
for age in age_counts.index:
    num_images_needed = int(target - age_counts[age]) if age_counts[age] < target else 0
    images_to_augment = full_dataset[full_dataset['age'] == age]

    for i in range(num_images_needed):
        original_image_row = images_to_augment.sample(n=1).iloc[0]
        original_image_path = os.path.join('../faces-age', original_image_row['filename'])
        image = Image.open(original_image_path).convert('RGB')

        # Perform augmentation
        augmented_image = augment_transforms(image)

        # Define new filename and save the augmented image
        new_filename = f'augmented_{original_image_row["id"]}_{i}.jpg'
        augmented_image_path = os.path.join(augmented_root_dir, new_filename)
        augmented_image.save(augmented_image_path)

        # Record the augmented image details
        augmented_rows.append({
            'id': full_dataset['id'].max() + 1 + i,
            'filename': new_filename,
            'age': age
        })

# Create a DataFrame with the augmented image details
augmented_data_df = pd.DataFrame(augmented_rows)

# Concatenate the original data with the augmented data
all_data_df = pd.concat([full_dataset, augmented_data_df], ignore_index=True)

# Save all image data to a new CSV file
all_data_csv_path = '../faces-age/all_data.csv'
all_data_df.to_csv(all_data_csv_path, index=False)
