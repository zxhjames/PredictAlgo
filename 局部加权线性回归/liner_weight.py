'''
Author: your name
Date: 2021-02-02 13:54:20
LastEditTime: 2021-02-02 18:55:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Code/PyCode/project_demo/国家电网电力预测/其他/局部加权线性回归/liner_weight.py
'''
'''
Author: your name
Date: 2021-02-02 13:42:50
LastEditTime: 2021-02-02 13:47:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Code/PyCode/project_demo/国家电网电力预测/其他/局部加权线性回归.py
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
def load_data(name):
    data_train = pd.read_csv(name)
    return data_train


def get_weights_iteration(X, x_test, k=1.0):
    diff = X - x_test
    temp = np.zeros(diff.shape[0])

    for i in range(temp.shape[0]):
        temp[i] = diff[i].dot(diff[i])

    w = np.exp(-temp / (2 * k**2))
    return w


def run_steep_gradient_descent(X, y, alpha, theta, weights):
    prod = (np.dot(X, theta) - y) * weights
    sum_grad = np.dot(prod, X)
    theta = theta - (alpha / X.shape[0]) * sum_grad
    return theta


def sum_of_square_error(X, y, theta, weights):
    prod = (np.dot(X, theta) - y) * weights
    error = (np.square(prod)).mean()
    return error


def local_weighted_linear_reg(X, y, x_test, iterations = 70000, alpha = 0.000001):
    theta = np.zeros(X.shape[1])
    weights = get_weights_iteration(X, x_test, 1.0)

    start = time.time()
    for i in range(iterations):
        theta = run_steep_gradient_descent(X, y, alpha, theta, weights)
        # error = sum_of_square_error(X, y, theta, weights)
        # print('At Iteration %d - Error is %.5f ' % (i + 1, error))
    time_temp = time.time() - start
    print('本次所花时间：%f秒' % time_temp)

    return theta


def main():
    # 数据预处理
    fp = open('l.csv','w',encoding='utf-8')
    csv_writer = csv.writer(fp)
    csv_writer.writerow(['局部加权线性回归'])

    total = 103
    step = 60
    for k in range(0,total):
        file = '../clear.xlsx'
        predict_path = 'predict.csv' 
        num = 2
        data = pd.read_excel(file,encoding='utf-8',sheet_name=k)
        data_predict = load_data(predict_path)
        pred = data.loc[len(data) - num + 1 : , :]
        data = data.loc[len(data) - step :len(data) - num,:]
       # print(data)
       # print(pred)
    
    # x_label = pd.to_datetime(x_label, infer_datetime_format=True)
        data['日期'] = pd.to_datetime(data['日期'], infer_datetime_format=True)
        data_predict['date'] = pd.to_datetime(data_predict['date'], infer_datetime_format=True)

        X = np.array((data['日期'] - data['日期'].min()).dt.days + 1)
        y = np.array(data['用电总和'])
        m = X.shape[0]
        ones = np.ones(m)
        X = np.vstack((X, ones)).T

        X_t = np.array((data_predict['date'] - data['日期'].min()).dt.days + 1)
        m_t = X_t.shape[0]

        ones = np.ones(m_t)
        X_t = np.vstack((X_t, ones)).T

        result = np.zeros(m_t)
        for i in range(m_t):
            print('正在跑第%d个数据...' % (i + 1))
            theta = local_weighted_linear_reg(X, y, X_t[i])
            result[i] = X_t[i].dot(theta)

        data_predict['y'] = result
        e = (data_predict['y'][0] - pred['用电总和']) / pred['用电总和']
        csv_writer.writerow(e)
    
    fp.close()
    print('end')
   # print(data_predict['y'][0] )
    #print(pred['用电总和'])
    #e = （data_predict['y'】[0] - pred['用电总和']）/ pred['用电总和']
    # data_predict.to_csv(predict_path, index=False)

    # y_predict = np.zeros(m)
    # for i in range(m):
    #     print('正在跑第%d个样本...' % (i + 1))
    #     theta = local_weighted_linear_reg(X, y, X[i])
    #     y_predict[i] = X[i].dot(theta)

    # y_predict = result

    # x_label = data_predict['date']

    # plt.figure(figsize=(18, 6))
    # plt.xlabel("date")
    # plt.ylabel('Net Asset Value')
    # plt.plot(x_label, y_predict, c = 'b')
    # # plt.scatter(x_label, y, c = 'r', marker='o')
    # plt.savefig('predict')
    # plt.show()


if __name__ == '__main__':
    main()