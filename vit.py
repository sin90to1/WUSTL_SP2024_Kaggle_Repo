import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

full_dataset = pd.read_csv('../faces-age/train.csv')
train_df, valid_df = train_test_split(full_dataset, test_size=0.2, random_state=42)  # 20% for validation
train_df.to_csv('../faces-age/train_split.csv', index=False)
valid_df.to_csv('../faces-age/valid_split.csv', index=False)

class FacesAgeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool, optional): Indicator if the dataset includes ages (default: True).
        """
        self.faces_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.faces_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.faces_frame.iloc[idx, 1])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        if self.train:
            age = self.faces_frame.iloc[idx, 2]
            return image, torch.tensor(age, dtype=torch.float)
        else:
            return image

# Note: For the test set, set train=False

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the model's input requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize datasets with the appropriate CSV files and root directory
train_dataset = FacesAgeDataset(csv_file='../faces-age/train_split.csv', root_dir='../faces-age', transform=transform)
valid_dataset = FacesAgeDataset(csv_file='../faces-age/valid_split.csv', root_dir='../faces-age', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)



import torch
from torchvision.models import vit_b_16

# Initialize the model
model = vit_b_16(pretrained=True)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 1)  # Adjust for regression

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


import torch.optim as optim
import torch.nn as nn

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

# Training and validation loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0
    for images, ages in train_loader:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}')
    
    # Validation phase
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, ages in valid_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            loss = criterion(outputs, ages.view(-1, 1))
            valid_loss += loss.item() * images.size(0)

    print(f'Epoch {epoch+1}, Validation Loss: {valid_loss / len(valid_loader.dataset):.4f}')
    
    
    # Define test set path and loader
test_csv_file = '../faces-age/test.csv'
test_root_dir = '../faces-age'

test_dataset = FacesAgeDataset(csv_file=test_csv_file, root_dir=test_root_dir, transform=transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Create DataLoader for test data similar to above

model.eval()
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.cpu().numpy())

# Prepare submission file
submission = pd.DataFrame({
    'id': range(18000, 18000 + len(predictions)),  # Adjust based on actual test CSV file
    'age': [float(pred[0]) for pred in predictions]
})
submission.to_csv('submission.csv', index=False)