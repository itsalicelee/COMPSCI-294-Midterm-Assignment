from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import Bunch

def classify(dataset, name):
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # First algorithm
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.title("{} Decision Tree Classifier\nAcc: {:.3f}".format(name, acc), fontsize = 28)
    plt.savefig('{}.1.png'.format(name.strip().lower()))
    print('Decision tree acc: {}'.format(acc))

    # Second algorithm
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.title("{} Decision Tree Classifier with max_depth=2\nAcc: {:.3f}".format(name, acc), fontsize=28)
    plt.savefig('{}.2.png'.format(name.strip().lower()))
    print('Decision tree (max depth=2) acc: {}'.format(acc))

if __name__ == '__main__':
    classify(load_breast_cancer(as_frame=True), 'Breast Cancer')
    classify(load_digits(as_frame=True), 'Digits')
    # random dataset 1
    data = np.random.rand(200,15)
    target = np.random.randint(2, size=data.shape[0])
    dataset = Bunch(data=pd.DataFrame(data), target=target)
    classify(dataset, 'Random 1 Dataset (200 samples, 15 features)')
    # random dataset 2
    data = np.random.rand(200,5)
    target = np.random.randint(2, size=data.shape[0])
    dataset = Bunch(data=pd.DataFrame(data), target=target)
    classify(dataset, 'Random 2 Dataset (200 samples, 5 features)')
    # random dataset 3
    data = np.random.rand(20, 15)
    target = np.random.randint(2, size=data.shape[0])
    dataset = Bunch(data=pd.DataFrame(data), target=target)
    classify(dataset, 'Random 3 Dataset (20 features, 15 samples)')
    