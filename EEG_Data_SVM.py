import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("eeg_data.csv", usecols = [ i for i in range(1,180)])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1], test_size=0.3)


svm_model = svm.SVC().fit(X_train, y_train) 

predictions = svm_model.predict(X_test)

print(classification_report(y_test, predictions, labels=[1,2,3,4,5]))



knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
predictions = knn.predict(X_test)
print(classification_report(y_test, predictions, labels=[1,2,3,4,5]))
