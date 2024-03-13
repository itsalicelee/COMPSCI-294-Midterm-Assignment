import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
        )
		
		self.flatten = nn.Flatten()
		self.fc = nn.Sequential(
			nn.Linear(64 * 3 * 3, 64),
			nn.ReLU(),
			nn.Linear(64, 10)		
		)

	def forward(self, x):
		x = self.cnn(x)
		x = self.flatten(x)
		x = self.fc(x)
		return x

def train(epochs):
	for epoch in range(epochs):
		for inputs, labels in tqdm(train_loader):
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

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

	model = Classifier()
	criterion = nn.CrossEntropyLoss()
	lr = 0.0001
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# Training the model
	print("Training the model")
	epochs = 5
	train(epochs=epochs)
	
	# Evaluate the model on the test set
	print("Evaluating the model")
	all_predictions, all_labels = eval()
	accuracy = accuracy_score(all_labels, all_predictions)
	print(f"Test Accuracy: {accuracy}")
