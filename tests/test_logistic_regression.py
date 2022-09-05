import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from uzapy.regression import LogisticRegression
from uzapy.utils import *

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LogisticRegression(0.00001)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy(y_test, predictions))