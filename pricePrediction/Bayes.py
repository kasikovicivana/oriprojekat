import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
import hickle as hkl

data = hkl.load('data.hkl')
X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']


NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
NB_pred = NB_model.predict(X_test)
NB_pred_train = NB_model.predict(X_train)
#NB_score = NB_model.score(X_test, y_test)
#NB_score_train = NB_model.score(X_train, y_train)
NB_matrix = confusion_matrix(y_test, NB_pred)
NB_matrix_train = confusion_matrix(y_train, NB_pred_train)
print("Trenirajuci skup: ", classification_report(y_train, NB_pred_train))
print("Testirajuci skup: ", classification_report(y_test, NB_pred))
plt.figure(figsize=(10, 7))
sns.heatmap(NB_matrix_train, annot=True)
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(NB_matrix, annot=True)
plt.show()

