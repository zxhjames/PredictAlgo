'''
Author: your name
Date: 2021-02-01 21:08:04
LastEditTime: 2021-02-02 20:17:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Code/PyCode/project_demo/国家电网电力预测/其他/client.py
'''
# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import csv

from numpy.core.records import format_parser
txt = "./prophet算法误差.txt"
 # fp = open(txt,"w+")
total = 103
step = 60
futureday = 3
# 写入excel文件
fp = open('t.csv','w',encoding='utf-8')
csv_writer = csv.writer(fp)
csv_writer.writerow(['prophet'])
# 读取文件
for k in range(0,total):
    df = pd.read_excel('./clear.xlsx',encoding='utf-8',sheet_name=k)
    name = df['用电企业名称'][0]
    print(name)
    newdf = pd.DataFrame()
    newdf['ds'] = df['日期']
    newdf['y'] = df['用电总和']
    # 筛选掉0值
    # newdf = newdf[~newdf['y'].isin([0])]

    # 每一列*1000
    # newdf['y'] = newdf['y'] 
    # newdf.to_csv('./newdf.csv')


    # 切分数据集
    l = len(newdf)
    print(l)
    train_data = (newdf.loc[l - step :l-futureday,:,])
    test_data = (newdf.loc[l-futureday:,:])
    m = Prophet(interval_width=0.8,changepoint_range=0.9,growth = "linear")
    m.fit(train_data)


    # 数据预测，periods表示预测点数，freq表示预测频率
    future = m.make_future_dataframe(periods=7,freq='D') # 预测未来
    forecast = m.predict(future)
    # m.plot(forecast) # 绘制图形
    l = len(forecast)

    # print(forecast)
    forecast = forecast[['trend']].loc[l-futureday:,:]
    test_data = (test_data[['y']])

    # error
    f = np.array(forecast)
    t = np.array(test_data)
    real = list()
    pred = list()
    err =  list()
    avg = 0.00
    count = 0
    for i in range(0,len(f)):
        real.append(t[i][0])
        pred.append(f[i][0])
        tmp = (f[i][0] - t[i][0]) / t[i][0]
        err.append(tmp)
        if t[i][0] != 0:
            avg = avg + tmp
            count = count + 1

    print(real)
    print(pred)
    print(err)
    csv_writer.writerow([err[0]])
    #fp.write('\n' + name + '\n')
    #fp.write('真实数据 => ' + str(real) + '\n')
    #fp.write('预测数据 => ' + str(pred) + '\n')
    #fp.write('相对误差 => ' + str(err)  + '\n')
    #fp.write('平均误差 => ' + str(avg / count) + '\n')
    
fp.close()
#fp.close() 
print('文件写入完毕')