# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import chardet and find the encoding of the dataset.

2.Import other necessary libraries and upload the csv file in the complier.

3.Find head,info and null elements of the dataset.

4.Using CounterVectorizer and SVC find the y prediction array and accuracy .

5.End the Program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 212222230092
RegisterNumber:  P NAVEEN KUAMR
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:
## 1. Result output
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/f6efb661-04e7-44e2-b84a-5cb3546fe49b)

## 2. data.head()
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/1262d5e5-d16c-43f6-8917-11f37dee54cb)

## 3. data.info()
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/866c81b0-531f-477d-9c6b-20beffcfba93)

## 4. data.isnull().sum()
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/b69d5420-7ab0-4829-8a3f-f0fdfa310c2a)

## 5. Y_prediction value
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/1f9a8053-fefb-4d5a-8f77-9182f45488a5)

## 6. Accuracy value
![image](https://github.com/Naveen22009215/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119401470/db0a28db-d3d4-459b-b8d3-6e5a3287453d)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
