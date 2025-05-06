# File: cnn.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import alexnet, AlexNet_Weights
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score
from PIL import Image, ExifTags
import os
from tqdm import tqdm
import multiprocessing
import numpy as np

# Path to the project directory
DATA_ROOT = "/Users/nathan/PycharmProjects/CS 422 Project/.venv"

# Custom AlexNet-based model for reduced classification
class Places365Reduced(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        for param in base_model.parameters():
            param.requires_grad = False
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Custom dataset loader for scene classification
class SceneDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())
            if exif.get(orientation) == 3:
                image = image.rotate(180, expand=True)
            elif exif.get(orientation) == 6:
                image = image.rotate(270, expand=True)
            elif exif.get(orientation) == 8:
                image = image.rotate(90, expand=True)
        except:
            pass

        if self.transform:
            image = self.transform(image)

        return image, label

# Load scene classification data for a building
def load_scene_data(building_prediction):
    image_paths, labels = [], []
    label_names = sorted(os.listdir(os.path.join(DATA_ROOT, building_prediction)))
    label_map = {label: i for i, label in enumerate(label_names)}

    for label in label_names:
        class_dir = os.path.join(DATA_ROOT, building_prediction, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label_map[label])

    return image_paths, labels, label_names

# Train model and evaluate validation accuracy
def train_and_evaluate(model, image_paths, labels, class_names, transform, device):
    dataset = SceneDataset(image_paths, labels, transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    label_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1. / torch.tensor(label_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=multiprocessing.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=multiprocessing.cpu_count())

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

    # Evaluate on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")

# Convenience wrapper to run end-to-end CNN training for a building
def run_scene_classifier(building_prediction):
    image_paths, labels, class_names = load_scene_data(building_prediction)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = AlexNet_Weights.IMAGENET1K_V1
    base_model = alexnet(weights=weights)
    model = Places365Reduced(base_model, len(class_names)).to(device)

    train_and_evaluate(model, image_paths, labels, class_names, transform, device)
