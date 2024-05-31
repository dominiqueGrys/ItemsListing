import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Add the yolov5 repository to the system path
sys.path.append(str(Path.cwd() / 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import LoadImages, LoadStreams, letterbox
from utils.torch_utils import select_device

# Load the pretrained YOLO model
device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device, dnn=False)

# Extract feature dimension from the penultimate layer's cv3 Conv2d layer
penultimate_layer = model.model.model[-2]
feature_dim = penultimate_layer.cv3.conv.out_channels

# Create a feed-forward network for counting
class TileCounter(nn.Module):
    def __init__(self, input_dim):
        super(TileCounter, self).__init__()
        self.fc = nn.Linear(input_dim, 1000)  # Assuming a max of 999 objects
        
    def forward(self, x):
        return self.fc(x)
        
counter = TileCounter(feature_dim)

# Define a hook to capture features
features = []

def hook(module, input, output):
    features.append(output)

# Register the hook
hook_handle = penultimate_layer.register_forward_hook(hook)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(counter.parameters(), lr=0.001)

def train_model(model, counter, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    model.eval()  # Set the YOLO model to evaluation mode
    counter.train()  # Set the counter model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Clear features list
            features.clear()
            
            # Forward pass through the YOLO model
            with torch.no_grad():
                _ = model(inputs)
                features_tensor = features[0].view(inputs.size(0), -1)
            
            print(f'Feature tensor shape: {features_tensor.shape}')
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = counter(features_tensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        validate_model(model, counter, val_loader, criterion)

    print('Finished Training')

def validate_model(model, counter, val_loader, criterion):
    model.eval()  # Set both models to evaluation mode
    counter.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Clear features list
            features.clear()
            
            # Forward pass through the YOLO model
            _ = model(inputs)
            features_tensor = features[0].view(inputs.size(0), -1)
            
            # Forward pass through the counter network
            outputs = counter(features_tensor)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%')


class RoofTileDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Convert to RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def list_files_in_directory(directory, extensions=['.png']):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files


# Example usage:
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

dir_path = os.path.dirname(os.path.realpath(__file__))
image_directory = f'{dir_path}/train/t/'

image_paths = list_files_in_directory(image_directory)

labels = [x for x in range(1, len(image_paths)+1)]  # Corresponding counts of roof tiles


# Split the dataset into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)


# Create dataset instances
train_dataset = RoofTileDataset(train_paths, train_labels, transform=transform)
val_dataset = RoofTileDataset(val_paths, val_labels, transform=transform)

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Determine the feature tensor shape by running one forward pass
model.eval()
counter.eval()
with torch.no_grad():
    features.clear()
    dummy_input = torch.zeros((1, 3, 640, 640)).to(device)
    _ = model(dummy_input)
    features_tensor = features[0].view(1, -1)
    feature_dim = features_tensor.shape[1]

counter = TileCounter(feature_dim)

# Redefine optimizer with the updated counter network
optimizer = optim.Adam(counter.parameters(), lr=0.001)

num_epochs = 25
train_model(model, counter, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=num_epochs)

# Load test dataset
test_image_directory = f"{image_directory}test/"
test_image_paths = list_files_in_directory(test_image_directory)
test_labels = [23]  # Corresponding counts of roof tiles in the test set

# Create dataset and dataloader for the test set
test_dataset = RoofTileDataset(test_image_paths, test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the test function
def test_model(model, counter, test_loader, criterion):
    model.eval()  # Set both models to evaluation mode
    counter.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Clear features list
            features.clear()
            
            # Forward pass through the YOLO model
            _ = model(inputs)
            features_tensor = features[0].view(inputs.size(0), -1)
            
            # Forward pass through the counter network
            outputs = counter(features_tensor)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy}%')

# Test the model
test_model(model, counter, test_dataloader, criterion)

# Don't forget to remove the hook when done
hook_handle.remove()

