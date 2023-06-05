# EXPERIMENT 09: IMPLEMENTAION OF SVM FOR SPAM MAIL DETECTION
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## EQUIPMENT'S REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1. Import the required packages.
2. Split data into training set and testing set.
3. Use CountVectorizer to extract features.
4. Import SVC and predict y values.
5. Find the accuracy of model.

## PROGRAM:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.Nithish
RegisterNumber:  212220040103
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv')
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
x_test=cv.fit_transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## OUTPUT:
### data.head():
![image](https://github.com/Rithigasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427256/ca3bff0f-8310-4993-833f-1ad9520b411a)

### data.info():
![image](https://github.com/Rithigasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427256/2dcffb92-a90c-4182-9a95-d18c24fd9be7)

### data.isnull().sum():
![image](https://github.com/Rithigasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427256/26667448-be8f-4524-8d60-0dbfb474f6b0)

### Y_prediction value:
![image](https://github.com/Rithigasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427256/71a5e79d-14e7-4456-bfd0-ba1e090c7a9d)

### Accuracy value:
![image](https://github.com/Rithigasri/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427256/f5392d62-348e-46a2-bbec-b93e4da3c096)


## RESULT:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
