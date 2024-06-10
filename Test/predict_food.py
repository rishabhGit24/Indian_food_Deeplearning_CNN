import torch
import torch.nn as nn
from torchvision.transforms import Resize, ToTensor
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# Define the device to use (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
model = resnet18(pretrained=False)
num_classes = 7
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('food_classifier.pth'))
model.to(device)
model.eval()

# Define the class labels
class_labels = ['aloo_paratha', 'biryani', 'burger', 'dosa', 'fried-rice', 'idly', 'vada']

# Load and preprocess the test image
test_image = Image.open(r"R:\CODE\CC_ML\food_jpg\istockphoto-1418100746-612x612.jpg")  # Use raw string literal (r) to handle backslashes
transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor()
])
test_tensor = transform(test_image).unsqueeze(0).to(device)

# Make the prediction
with torch.no_grad():
    outputs = model(test_tensor)
    _, predicted_class = torch.max(outputs, 1)
    predicted_label = class_labels[predicted_class.item()]

print(f"The predicted class is: {predicted_label}")
