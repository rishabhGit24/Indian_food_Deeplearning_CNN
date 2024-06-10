import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Define the device to use (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms for the dataset
transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor()
])

# Load the datasets
train_dataset = ImageFolder("R:\CODE\CC_ML\delicacies", transform=transform)
val_dataset = ImageFolder("R:\CODE\CC_ML\delicacies", transform=transform)
test_dataset = ImageFolder("R:\CODE\CC_ML\Test", transform=transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
model = resnet18(pretrained=True)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_accuracy += torch.sum(preds == labels).item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
test_accuracy = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        test_accuracy += torch.sum(preds == labels).item()
test_loss /= len(test_loader)
test_accuracy /= len(test_dataset)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'food_classifier.pth')
