import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a=pd.read_csv('seattle-weather.csv')
a
a=a.drop(['date'],axis=1)
a
from sklearn.preprocessing import LabelEncoder
a['weather'].value_counts()
l=LabelEncoder()
a['w']=l.fit_transform(a['weather'])
a=a.drop(['weather'],axis=1)
a
from sklearn.model_selection import train_test_split
a.columns
x=a[['precipitation', 'temp_max', 'temp_min', 'wind']]
y=a['w']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)
from sklearn.linear_model import LinearRegression
m=LinearRegression()
m.fit(x_train,y_train)
m.predict([[8.6,4.4,1.7,1.3]])
y_predict=m.predict(x_test).round(0)
y_test=y_test.round(0)
from sklearn.metrics import f1_score
f1_score(y_test,y_predict,average='micro')                                                                                             
i=np.array(range(50))
plt.scatter(i,y_predict[0:50])
plt.scatter(i,y_test[0:50])
                                               
                                               