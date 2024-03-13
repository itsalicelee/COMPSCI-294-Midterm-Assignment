from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the binary classification breast cancer dataset
dataset = load_digits()
data, target = dataset.data, dataset.target
# train test split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# K Nearest Neighbors
bits_per_parameter = []
for k in range(1, 1000):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc = knn.score(X_test, y_test)

    res = 1 / acc
    if res > 2:
        assert False
    bits_per_parameter.append(res)


# plt.figure(figsize=(8, 6))
# plt.plot(range(1,21), bits_per_parameter)
# plt.xticks(range(1,21))
# plt.ylabel('Information limit per parameter')
# plt.xlabel('Number of Neighbors (k)')
# plt.title('Information limit per parameter vs Number of Neighbors (k) for KNN Classifier')
# plt.savefig('6.1.b.png', dpi=300)
# plt.show()