'''
Author: your name
Date: 2021-02-01 16:23:05
LastEditTime: 2021-02-01 16:28:47
LastEditors: your name
Description: In User Settings Edit
FilePath: /Code/PyCode/project_demo/remotegit/LSTM/myAlgo/prophet.py
'''
from fbprophet import Prophet
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
plt.style.use('bmh')
future_days = 10;


df = pd.read_csv('./stocktest.csv')
df.head(10)

Y = np.array(df['Sales'])[:-future_days]
print(Y)

model = Prophet(interval_width=0.95)
model.fit(Y)

# 数据预测，periods表示预测点数，freq表示预测频率
future = model.make_future_dataframe(periods=3,freq='YS') # 预测未来
forecast = model.predict(future_days)

model.plot(forecast) # 绘制图形