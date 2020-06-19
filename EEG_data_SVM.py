from scipy.io import loadmat
import numpy as np
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm


d = None
l = None
z = 0
for s in os.listdir('.'):
 if os.path.isfile(s):
  continue
 print(s)
 for t in os.listdir(s):
  x = loadmat("/".join([s,t]))['tc_rest_aal'].flatten()
  x = x.reshape(1,x.size)
  d = np.vstack((d,x)) if not (d is None) else x
  l = np.append(l,z) if not (l is None) else z
 z += 1

X_train, X_test, y_train, y_test = train_test_split(d, l, test_size=0.3)


clf = svm.SVC(C=2)
clf.fit(X_train, y_train)
res = clf.predict(X_test)
print(classification_report(y_test, res, labels=[0,1]))
print(res)
print(y_test)




%{
cvx_begin
  variables x(10);
  maximize( sum(entr(x)));
  subject to 
   ones(1,10) * x == 1
cvx_end
x
scatter(1:1:10, x)
xticks(1:1:10)
%}



cvx_begin
 variables x(20) p(20);
 minimize 1
 subject to 
  ones(1,20) * p == 1
  p >= 0
  -0.1 <= x'*p <= 0.1  
cvx_end
cvx_optval
cvx_status
