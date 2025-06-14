import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv(r"C:\Users\Dell\Desktop\Admission_predictor\Admission_Predict_Ver1.csv")


df.head()
df.keys()

# Drop the 'Serial No.' column
df = df.drop(columns=['Serial No.'])

df.shape

df.info()

df.describe()

df.columns

y =df['Chance of Admit ']

x =df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']]

x.shape,y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=10)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.linear_model import LinearRegression
model =LinearRegression()

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

print("y_pred",y_pred)

print("y_test",y_test)

#calculate the accuracy of the model
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

print(mean_absolute_error(y_test,y_pred))

mean_p=mean_absolute_percentage_error(y_test,y_pred)

print("mean_p",mean_p)

Accuracy=( 1-mean_p) * 100

print("Accuracy",Accuracy)

import joblib

joblib.dump(model, "linear_model2.joblib")
print("Model saved to linear_model2.joblib")





