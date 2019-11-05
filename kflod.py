import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\VIPUL\\Downloads\\kflod\\Purchased_Dataset.csv")
x = df[['Age','EstimatedSalary']]
y = df['Purchased']
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state =5)
knnclassifier = KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(x_train,y_train)
y_pred = knnclassifier.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
from sklearn.model_selection import cross_val_score
knnclassifier = KNeighborsClassifier(n_neighbors=4)
print(cross_val_score(knnclassifier, x, y, cv=10, scoring ='accuracy').mean())