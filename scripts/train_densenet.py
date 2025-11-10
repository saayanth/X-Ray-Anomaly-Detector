import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ----------------------------
# ✅ Configuration
# ----------------------------
DATA_DIR = r"C:\Users\PC\xray_project\dataset\balanced_dataset"
MODEL_SAVE_PATH = r"C:\Users\PC\xray_project\models\densenet_balanced.pth"
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_CLASSES = len(os.listdir(DATA_DIR))  # Automatically detect class count

# ----------------------------
# ✅ Data preprocessing
# ----------------------------
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Split dataset automatically
train_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms["train"])
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# ✅ Model setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(weights="IMAGENET1K_V1")

# Replace final layer
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# ✅ Training loop
# ----------------------------
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    val_loss, val_acc = 0.0, 0.0

    # ----------------------------
    # ✅ Validation
    # ----------------------------
    model.eval()
    with torch.no_grad():
        correct_val, total_val = 0, 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

        val_acc = correct_val / total_val

    print(f"Train Loss: {train_loss/len(train_loader.dataset):.4f}, "
          f"Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss/len(val_loader.dataset):.4f}, "
          f"Val Acc: {val_acc:.4f}")

# ----------------------------
# ✅ Save trained model
# ----------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
