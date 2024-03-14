import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def computeMEC(data):
    mec = 0
    n_size = len(data)
    for i in range(10):
        p = data.count(i) / n_size
        mec += - data.count(i) * (np.log2(p))
    return mec 
    
# MEC  = 785*128 +128 + 64 + 10
class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.fc = nn.Sequential(
            
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# MEC  = 785*64 + 64 + 10
class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# MEC  = 785*256 + 256 + 64 + 10
class Classifier3(nn.Module):
    def __init__(self):
        super(Classifier3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(epochs, model, model_name):
    losses = []
    for epoch in range(epochs):
        l = []
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            l.append(loss.item())
        losses.append(np.mean(l))
    return losses         

def eval():
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_predictions, all_labels

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    targets = []
    for data, target in train_dataset:
        targets.append(target)
    
    mec = computeMEC(targets)
    
    model_losses = []
    model_names = ['Classifier1', 'Classifier2', 'Classifier3']
    models = [Classifier1(), Classifier2(), Classifier3()]
    for i in range(3):
        model = models[i]
        criterion = nn.CrossEntropyLoss()
        lr = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training the model
        print("Training Model{}".format(i + 1))
        epochs = 50
        loss = train(epochs, model, model_names[i])
        model_losses.append(loss)
    
        # Evaluate the model on the test set
        print("Evaluating Model{}".format(i + 1))
        all_predictions, all_labels = eval()
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Test Accuracy Model{}: {:.3f}".format(i + 1, accuracy))
loss1, loss2, loss3 = model_losses[0], model_losses[1], model_losses[2]
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), loss1, label='Classifier1')
plt.plot(range(epochs), loss2, label='Classifier2')
plt.plot(range(epochs), loss3, label='Classifier3')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs for three Models')
plt.legend()
plt.savefig('9.1.png')

    
