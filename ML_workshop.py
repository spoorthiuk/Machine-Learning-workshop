

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("titanic.csv")

df.head()
#df.tail()
#df.head(8)
#print(df.head())
#df.loc[100:130,'Survived':'Age']

df.shape

df.info()
#df.columns

#df.isna()
df.isna().sum()

df["Ticket"]
#df[(df["Survived"]==0)&(df["Gender"]== 'male')][["Survived","Gender"]]
#print information of female passengers

df.corr()
plt.figure(figsize=(7,7))
sns.heatmap(df.corr())
#sns.heatmap(df.corr(),annot=True,linewidths=0.3)

"""### **Creating new features**"""

#new column that classifies passengers under the age of 16 as child
def who_col(lis):
  age,gender=lis
  if age<16:
    return "child" #irrespective of the gender
  elif gender=="male":
    return "man"
  else:
    return "woman"

#adding new column called who
df["who"]=df[["Age","Gender"]].apply(who_col,axis=1)

df.head(8)

df.info()

#Encoding the values (child:0,man:1,woman:2)
enc1={'child':0,'man':1,'woman':2}
df['who']=df.who.map(enc1)
df.head()

"""### **Handling Null values**"""

##age is of the type float and has null values, fill them up with the mean value
m=df['Age'].mean()
#m
df['Age']=df['Age'].fillna(m)
df.info()

##Encoding Embarked

#df['Embarked'].unique()
enc2={'S':1,'C':2,'Q':3}
df['Embarked']=df['Embarked'].map(enc2)
df['Embarked']=df['Embarked'].fillna(0)
df.info()

##Encoding Cabin

df['Cabin']=df['Cabin'].str[0]
df['Cabin'].unique()
enc3={'C':1, 'E':2, 'G':3, 'D':4, 'A':5, 'B':6, 'F':7, 'T':8}
df['Cabin']=df['Cabin'].map(enc3)
df['Cabin']=df['Cabin'].fillna(0)
df.head()

df.info()

"""### **Data visualization**"""

df.corr()
plt.figure(figsize=(10,10))
#sns.heatmap(df.corr())
sns.heatmap(df.corr(),annot=True,linewidths=0.3)

sns.factorplot("Pclass", "Survived", data=df, hue="who")

sns.barplot("Cabin", "Survived", data=df)
#{'C':1, 'E':2, 'G':3, 'D':4, 'A':5, 'B':6, 'F':7, 'T':8}

#https://www.datacamp.com/community/tutorials/seaborn-python-tutorial
#https://www.w3schools.com/python/matplotlib_pyplot.asp
df.info()

"""### **Dropping Columns**"""

drop_list=['PassengerId','Name','Gender','Ticket']
df2=df.drop(drop_list,axis=1)

df2.head()
df2.shape

X=df2.drop('Survived',axis=1)
Y=df2['Survived']

X.head()

Y.head()

"""### **Training a model**"""

from sklearn.model_selection import train_test_split
 
# splitting data in training set(70%) and test set(30%).
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
 
lr = LogisticRegression() #create the object of the model
lr = lr.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
 
act = accuracy_score(y_train,lr.predict(x_train))
print('Training Accuracy is: ',(act*100))
p = precision_score(y_train,lr.predict(x_train))
print('Training Precision is: ',(p*100))
r = recall_score(y_train,lr.predict(x_train))
print('Training Recall is: ',(r*100))
f = f1_score(y_train,lr.predict(x_train))
print('Training F1 Score is: ',(f*100))

act = accuracy_score(y_test,lr.predict(x_test))
print('Test Accuracy is: ',(act*100))
p = precision_score(y_test,lr.predict(x_test))
print('Test Precision is: ',(p*100))
r = recall_score(y_test,lr.predict(x_test))
print('Test Recall is: ',(r*100))
f = f1_score(y_test,lr.predict(x_test))
print('Test F1 Score is: ',(f*100))

##Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 3, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features=0.5, 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)
rf = rf.fit(x_train,y_train)

act = accuracy_score(y_train,rf.predict(x_train))
print('Training Accuracy is: ',(act*100))
p = precision_score(y_train,rf.predict(x_train))
print('Training Precision is: ',(p*100))
r = recall_score(y_train,rf.predict(x_train))
print('Training Recall is: ',(r*100))
f = f1_score(y_train,rf.predict(x_train))
print('Training F1 Score is: ',(f*100))

act = accuracy_score(y_test,rf.predict(x_test))
print('Test Accuracy is: ',(act*100))
p = precision_score(y_test,rf.predict(x_test))
print('Test Precision is: ',(p*100))
r = recall_score(y_test,rf.predict(x_test))
print('Test Recall is: ',(r*100))
f = f1_score(y_test,rf.predict(x_test))
print('Test F1 Score is: ',(f*100))

rf.predict(x_train.head(8))

y_train.head(8)

#Workshop resources : https://drive.google.com/folderview?id=1eUanb2doYGZ4B5j2SQJM0JlqqdJJTR2b
#kaggle : https://www.kaggle.com/c/titanic
#seaborn : https://www.datacamp.com/community/tutorials/seaborn-python-tutorial
#matplotlib : https://www.w3schools.com/python/matplotlib_pyplot.asp
#scikit : https://www.tutorialspoint.com/scikit_learn/index.htm