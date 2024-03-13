from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

def classify(dataset, name):
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # First algorithm
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.title("{} Decision Tree Classifier".format(name), fontsize = 28)
    plt.savefig('{}.1.png'.format(name.strip().lower()))
    print('Decision tree acc: {}'.format(acc))

    # Second algorithm
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.title("{} Decision Tree Classifier with max_depth=2".format(name), fontsize=28)
    plt.savefig('{}.2.png'.format(name.strip().lower()))
    print('Decision tree (max depth=2) acc: {}'.format(acc))

if __name__ == '__main__':
    classify(load_breast_cancer(), 'Breast Cancer')
    classify(load_digits(), 'Digits')