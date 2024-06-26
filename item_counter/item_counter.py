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


import cv2
import numpy as np
import matplotlib.pyplot as plt



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
        self.fc = nn.Linear(input_dim, 1)  # Output a single number for counting
        
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
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
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
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = counter(features_tensor).squeeze()
            loss = criterion(outputs, labels.float())
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
            outputs = counter(features_tensor).squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            
            # Check accuracy (rounded prediction to the nearest integer)
            predicted = torch.round(outputs)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
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
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, filename))
    return files


# Example usage:
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

dir_path = os.path.dirname(os.path.realpath(__file__))
image_directory = f'{dir_path}/train/t2/'

image_paths = list_files_in_directory(image_directory)

#labels = [x for x in range(1, len(image_paths)+1)]  # Corresponding counts of roof tiles
labels = [1,2,2,2,3,4,5,7,8,7 ]

print(f"{image_paths} {labels}")

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
test_labels = [8,5]  # Corresponding counts of roof tiles in the test set

# Create dataset and dataloader for the test set
test_dataset = RoofTileDataset(test_image_paths, test_labels, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the test function
def test_model_old(model, counter, test_loader, criterion):
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
            outputs = counter(features_tensor).squeeze()
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            
            # Check accuracy (rounded prediction to the nearest integer)
            predicted = torch.round(outputs)
            print(f"prediction: {predicted}")
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy}%')




def test_model(model, counter, test_loader, criterion, save_path='output/'):
    model.eval()  # Set both models to evaluation mode
    counter.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    conf_thres = 0.05
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # Clear features list
            features.clear()
            
            # Forward pass through the YOLO model to get features and predictions
            predictions = model(inputs)
            features_tensor = features[0].view(inputs.size(0), -1)
            
            # Forward pass through the counter network
            outputs = counter(features_tensor).squeeze()
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            
            # Check accuracy (rounded prediction to the nearest integer)
            predicted = torch.round(outputs)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            print(f"predicted: {predicted}")
            
            # Apply non-maximum suppression to filter boxes
            predictions = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=0.15)
            
            for j in range(inputs.size(0)):
                img = inputs[j].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if predictions[j] is not None:
                    boxes = predictions[j][:, :4]
                    scores = predictions[j][:, 4]
                    
                    for box, score in zip(boxes, scores):
                        if score > conf_thres:  # Draw only if the confidence score is above 0.25
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                # Save the image
                cv2.imwrite(f'{save_path}/result_{i}_{j}.jpg', img)
    
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy}%')





# Test the model
test_model(model, counter, test_dataloader, criterion)

# Don't forget to remove the hook when done
hook_handle.remove()

