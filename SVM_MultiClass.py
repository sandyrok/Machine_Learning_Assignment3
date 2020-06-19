import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv('board_data_25.txt', header = None)
dfr, dfs, yr, ys = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.23)

svm_model_linear = SVC().fit(dfr, yr) 
predictions = svm_model_linear.predict(dfs) 
accuracy = svm_model_linear.score(dfs, ys) 
"""
knn = KNeighborsClassifier(n_neighbors = 7).fit(dfr, yr)
accuracy = knn.score(dfs, ys)
predictions = knn.predict(dfs)  
cm = confusion_matrix(ys, predictions)   


gnb = GaussianNB().fit(dfr,yr) 
predictions = gnb.predict(dfs) 
accuracy = gnb.score(dfs, ys) 
"""

clf = BaggingClassifier(base_estimator= KNeighborsClassifier(n_neighbors = 7), n_estimators=10, random_state=0).fit(dfr, yr)
predictions = clf.predict(dfs)

#print(accuracy)
cm = confusion_matrix(ys, predictions)
print(classification_report(ys, predictions, labels=[0,1,2,3]))
print(cm)
