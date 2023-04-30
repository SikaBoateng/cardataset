import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data =pd.read_csv(r'C:\Users\USER\Desktop\machine learning\knnclassifier\car.data')
data1= pd.DataFrame(data)
# print(data.head())

X = data1[['buying','maint','safety']].values
y = data1[['class']].values.ravel()

# print(X,y)
# print(len(X[0]))


# converting X into data 
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

# print(X)

# converting y into numerical values
# Label_mapping = {
#     'unacc': 0,
#     'acc':1,
#     'good':2,
#     'vgood':3
# }
# y.loc[:, 'class']= y['class'].map(Label_mapping).astype(int)
y = Le.fit_transform(y)


# print(y)

knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=1000)
# print(y_train.isna().sum())

# check for unique values in y_train
# print(y_train.unique())

knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("accuracy" , accuracy)