import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn

# Reading the dataset and splitting into train and validation sets
full_dataset = pd.read_csv('../faces-age/data_augmented/all_data.csv')
train_df, valid_df = train_test_split(full_dataset, test_size=0.2, random_state=42)
train_df.to_csv('../faces-age/data_augmented/train_split.csv', index=False)
valid_df.to_csv('../faces-age/data_augmented/valid_split.csv', index=False)

# Dataset class
class FacesAgeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train=True):
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

# Transformations with added augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data loaders
train_dataset = FacesAgeDataset(csv_file='../faces-age/data_augmented/train_split.csv', root_dir='../faces-age/data_augmented', transform=transform)
valid_dataset = FacesAgeDataset(csv_file='../faces-age/data_augmented/valid_split.csv', root_dir='../faces-age/data_augmented', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Model initialization
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),  # Adding dropout
    torch.nn.Linear(model.fc.in_features, 1)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)

# Early stopping setup
early_stopping_patience = 10
epochs_no_improve = 0
min_val_loss = float('inf')

# Training and validation loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, ages in train_loader:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages.view(-1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    # Validation phase
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, ages in valid_loader:
            images, ages = images.to(device), ages.to(device)
            outputs = model(images)
            loss = criterion(outputs, ages.view(-1, 1))
            valid_loss += loss.item() * images.size(0)

    scheduler.step(valid_loss / len(valid_loader.dataset))  # Scheduler step

    # Early stopping check
    if valid_loss < min_val_loss:
        min_val_loss = valid_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print('Early stopping!')
            break

    # Logging for tracking
    print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {valid_loss / len(valid_loader.dataset):.4f}')

