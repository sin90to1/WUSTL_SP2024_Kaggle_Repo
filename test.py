import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import os

# Dataset class for testing (without ages)
class FacesAgeTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.faces_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.faces_frame.iloc[idx, 1])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Transformation for the test set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = FacesAgeTestDataset(csv_file='../faces-age/test.csv', root_dir='../faces-age', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Reconstruct the trained model
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 1)
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Perform prediction
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.view(-1).cpu().numpy())

# Generate IDs based on the range provided in your example
ids = list(range(18000, 18000 + len(predictions)))

# Create DataFrame for submission
submission = pd.DataFrame({
    'id': ids,
    'age': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file has been created successfully!")
