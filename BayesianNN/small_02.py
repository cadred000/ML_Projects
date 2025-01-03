import matplotlib.pyplot as plt
import numpy as np
# import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import torchbnn as bnn
# %matplotlib inline

print(torch.cuda.is_available())
torch.cuda.manual_seed(42)

data = '/home/dggibso1/.cache/kagglehub/datasets/jiayuanchengala/aid-scene-classification-datasets/versions/1/AID'

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
        )
])

dataset = datasets.ImageFolder(root=data, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])


batch_size =  16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)

class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.bconv1 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.001, in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bconv2 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.001, in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.bconv3 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.001, in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.bl1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.001, in_features=360000, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bl2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.001, in_features=512, out_features=128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bl3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.001, in_features=128, out_features=30)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.bconv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.bconv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.bconv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn4(self.bl1(x)))
        x = self.relu(self.bn5(self.bl2(x)))
        x = self.bl3(x)
        return x


device = torch.device('cuda') 
model = BNN()

criterion = nn.CrossEntropyLoss()
kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-6)

num_epochs = 5 

for layer in model.children():
    if isinstance(layer, bnn.BayesLinear):  
        torch.nn.init.xavier_normal_(layer.weight_mu)  
        if hasattr(layer, 'weight_rho'):  
            torch.nn.init.constant_(layer.weight_rho, -0.5)  
    elif hasattr(layer, 'weight') and layer.weight.dim() >= 2:  
        torch.nn.init.xavier_normal_(layer.weight)

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    print(f"Computing epoch [{epoch + 1}/{num_epochs}]....")
    for data, target in train_loader:
        optimizer.zero_grad()  

        output = model(data)

        loss = criterion(output, target) + kl_loss_fn(model) * 0.001

        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 10.0:  
                print(f"High gradient norm for {name}: {param.grad.norm().item()}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    model.eval()  
    total_val_loss = 0.0
    correct = 0

    with torch.no_grad():  
        for data, target in val_loader:
            output = model(data)  

            loss = criterion(output, target) + kl_loss_fn(model) * 0.1
            total_val_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

model.eval()  
correct = 0
all_labels = []
all_preds = []

with torch.no_grad():  
    for data, target in test_loader:
        output = model(data)  
        pred = output.argmax(dim=1, keepdim=True)  
        all_preds.extend(pred.view(-1).cpu().numpy())
        all_labels.extend(target.view(-1).cpu().numpy())
        correct += pred.eq(target.view_as(pred)).sum().item()

test_accuracy = 100. * correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.2f}%")

class_names = dataset.classes

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Data")
plt.savefig("confusion_matrix.png")
plt.close()

