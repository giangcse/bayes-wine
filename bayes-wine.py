# Cau 1
import pandas as pd
import numpy as np
dt = pd.read_csv('F:\Study\CT202 - Machine Learning\winequality-red.csv', delimiter=';')
X = dt[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y = dt["quality"]

# Cau 2
from sklearn.model_selection import KFold
kf = KFold(n_splits=50, shuffle=True)
for train_index, test_index in kf.split(X): #split()
  print('Train: ', train_index, 'Test: ', test_index) #In gia tri chi so cua tap huan luyen va tap kiemtra
  X_train, X_test = X.ix[train_index], X.ix[test_index] #Tao bien X_train, X_test de luu truthuoc tinh cua tap train va test
  y_train, y_test = y.ix[train_index], y.ix[test_index] #Tao bien y_train, y_test de luu tru nhancua tap train va test
  print('X_test: ', X_test)
  print '================'
  
# Cau 3
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# model = GaussianNB()
# model.fit(X_train, y_train)
# print model
# thucte = y_test
# dubao = model.predict(X_test)
# thucte
# dubao
# from sklearn.metrics import confusion_matrix
# cnf_matrix_gnb = confusion_matrix(thucte, dubao)
# print cnf_matrix_gnb
# Cau 4
