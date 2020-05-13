#Clasification of a breast cancer in malignant or benignant, implementing support vector machines (SVM)

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#The cancer caracteristics ['mean radius', 'mean texture', 'mean perimeter', ...]
#print(cancer.feature_names)
#print(cancer.target_names)

#data for every caracteristic in the cancer case
x = cancer.data

#Target ['benignant', 'malignant']
y = cancer.target

#test and train data split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["malignant", "benignant"]

#fill the virtual machine with train data
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

#clasify a tumour and obtain a target giving the model test data. 
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)

