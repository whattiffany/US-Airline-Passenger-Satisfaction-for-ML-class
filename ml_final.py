import numpy as np
import pandas as pd
import xlrd
import time
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from nn_model import nn_model

print('Start loading dataset:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
data = pd.read_excel('satisfaction.xlsx')
print('End loading dataset:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print(data.isnull().sum())

data['Arrival Delay in Minutes'].fillna(data['Arrival Delay in Minutes'].mode()[0], inplace=True)

data_frame = pd.DataFrame(data)
Gender = data_frame['Gender'].unique()
Customer = data_frame['Customer Type'].unique()
Class = data_frame['Class'].unique()
Satisfaction = data_frame['satisfaction_v2'].unique()
Type = data_frame['Type of Travel'].unique()

#LabelEncoder
LE = LabelEncoder()
GenderIndex = LE.fit_transform(Gender)
CustomerIndex = LE.fit_transform(Customer)
ClassIndex = LE.fit_transform(Class)
SatisfactionIndex = LE.fit_transform(Satisfaction)
TypeIndex = LE.fit_transform(Type)

def replaceDataPre(index,content,name):
  for i in range(0,len(content)):
    data_frame[name].replace(content[i],index[i],inplace=True)

replaceDataPre(GenderIndex,Gender,'Gender')
replaceDataPre(CustomerIndex,Customer,'Customer Type')
replaceDataPre(ClassIndex,Class,'Class')
replaceDataPre(SatisfactionIndex,Satisfaction,'satisfaction_v2')
replaceDataPre(TypeIndex,Type,'Type of Travel')

X = data_frame.drop(['id','satisfaction_v2'], axis=1)
y = data_frame['satisfaction_v2']
print(X.shape)

def prepare_data():
  #Principal component analysis 
  X_std = StandardScaler().fit_transform(X)
  pca = PCA(n_components=18)
  x_pca = pca.fit_transform(X_std)
  return pca,x_pca

pca,x_pca = prepare_data()

X_train, X_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2,random_state=99)
model = LogisticRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

import pickle
model_columns = list(pd.DataFrame(data_frame).drop(['id','satisfaction_v2'], axis=1).columns)
print('model_columns',model_columns)
pickle.dump(model_columns, open('model_columns.pkl','wb'))
pickle.dump(model, open('model.pkl','wb'))

# nn_model = nn_model(X_train,y_train,X_test,y_test)
# print('nn_model Evaluate:',nn_model)

print('Y_pred', y_pred)
print('accuracy:', accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
cm.transpose()[:-1,:-1]
print('confusion_matrix\n',cm)

print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))