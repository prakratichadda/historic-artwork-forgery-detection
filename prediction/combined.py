import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import seaborn as sns
import matplotlib.pyplot as plt

# ====== CONFIG ======
IMAGE_FOLDER = "data/historic-art/complete/artwork"
CSV_PATH = "data/mapped.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== LOAD AND CLEAN DATA ======
df = pd.read_csv(CSV_PATH)
df = df.head(1000)

# Keep only rows whose image file was actually downloaded
df = df[df['ID'].apply(lambda i: os.path.exists(os.path.join(IMAGE_FOLDER, f"{i}.jpg")))].reset_index(drop=True)
print(f"Images with files present: {len(df)}")

# Keep only top 15 artists, group rest as 'Others'
top_artists = df['artist'].value_counts().nlargest(15).index
df['artist'] = df['artist'].apply(lambda x: x if x in top_artists else 'Others')

# Filter rare classes
df = df.groupby('artist').filter(lambda x: len(x) > 1)

# Encode labels AFTER filtering
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['artist'])

import pickle
os.makedirs("models", exist_ok=True)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved label encoder to 'models/label_encoder.pkl'")

# ====== DATASET CLASS ======
class ArtDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.df = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['ID']
        label = row['label']
        image_path = os.path.join(self.image_folder, f"{image_id}.jpg")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# ====== TRANSFORMS ======
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====== SPLIT DATA ======
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = ArtDataset(train_df, IMAGE_FOLDER, transform=train_transform)
val_dataset = ArtDataset(val_df, IMAGE_FOLDER, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ====== MODEL SETUP ======
num_classes = df['label'].nunique()
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ====== TRAIN LOOP ======
best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(DEVICE)
        labels = labels.long().to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

    # ====== VALIDATION ======
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.long().to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / total
    print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_artist_classifier_cnn.pth")
        print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}")

print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

# ====== REPORT ======
unique_labels = sorted(set(all_labels + all_preds))
report = classification_report(
    all_labels,
    all_preds,
    labels=unique_labels,
    target_names=label_encoder.inverse_transform(unique_labels),
    zero_division=0
)
with open("cnn_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Classification report saved to 'cnn_classification_report.txt'")


# ====== CONFUSION MATRIX ======
cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=label_encoder.inverse_transform(unique_labels),
            yticklabels=label_encoder.inverse_transform(unique_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")
