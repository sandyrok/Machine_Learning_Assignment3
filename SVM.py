import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import SCORERS
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('data_noise_40.txt', header = None)
dfr, dfs, yr, ys = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.23)
"""
clf = svm.SVC()
clf.fit(dfr, yr)
res = clf.predict(dfs)
print(classification_report(ys, res, labels=[0,1]))


clf = svm.SVC( C=1, kernel = 'rbf')
scores = cross_validate(clf,df.iloc[:,:-1], df.iloc[:,-1], cv=8, scoring = ['precision', 'recall', 'f1'])
print( scores)

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(df.iloc[:,:-1], df.iloc[:,-1])
predicted_values = neigh.predict(dfs)
print(classification_report(ys, predicted_values, labels=[0,1]))
scores = cross_validate(clf,df.iloc[:,:-1], df.iloc[:,-1], cv=8, scoring = ['precision', 'recall', 'f1'])
print( scores)
"""

clf = AdaBoostClassifier( n_estimators=200, random_state=0).fit(dfr, yr)
predictions = clf.predict(dfs)
print(classification_report(ys, predictions, labels=[0,1]))
