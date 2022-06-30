from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import hickle as hkl

data = hkl.load('data.hkl')
X_train = data['xtrain']
X_test = data['xtest']
y_train = data['ytrain']
y_test = data['ytest']

best = None
acc = 0
k = 0
for i in list(range(1, 31)):
    KNN_model = KNeighborsClassifier(n_neighbors=i)
    KNN_model.fit(X_train, y_train)
    KNN_score = KNN_model.score(X_test, y_test)
    if KNN_score > acc:
        acc = KNN_score
        best = KNN_model
        k = i
print(f"Najbolji model je za k = {k}, sa tacnoscu {acc}")
KNN_pred = best.predict(X_test)
KNN_pred_train = best.predict(X_train)
KNN_matrix = confusion_matrix(y_test, KNN_pred)
KNN_matrix_train = confusion_matrix(y_train, KNN_pred_train)
print("Trenirajuci skup: ",classification_report(y_train, KNN_pred_train))
print("Testirajuci skup: ",classification_report(y_test, KNN_pred))
plt.figure(figsize=(10, 7))
sns.heatmap(KNN_matrix_train, annot=True)
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(KNN_matrix, annot=True)
plt.show()


