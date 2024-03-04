import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iteration = 10001
    n = len(x)
    learning_rate = 0.08

    for i in range(iteration):
        # y = m * x + b
        y_predicted = m_curr * x + b_curr
        # cost = 1/n * (y - y_predicted)**2
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        # md = -(2/n) * (x * (y - y_predicted)) derivative of cost with respect to the (x):
        md = -(2/n) * sum(x * (y - y_predicted))
        # bd = -(2/n) * (y - y_predicted) derivative of cost with respect to the (y):
        bd = -(2/n) * sum(y - y_predicted)
        # taking out the individual learning of data points:
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
plt.scatter(x=x, y=y, color='red')
plt.show()


# Appling the gradient algorithm to the find between the math and cs teat_score:
def gradient_descent1(x1, y1):
    m_curr1 = 0
    b_curr1 = 0
    iteration1 = 1000000
    n1 = len(x1)
    learning_rate1 = 0.0002

    for i in range(iteration1):
        # y = m * x1 + b
        y_predicted1 = m_curr1 * x1 + b_curr1
        # cost1 = (1/n1) * (y1 - y_predicted1)**2
        cost1 = (1/n1) * sum([val**2 for val in (y1 - y_predicted1)])
        # md1 = -(2/n1) * (x1 * (y1 - y_predicted1))
        md1 = -(2/n1) * sum(x1 * (y1 - y_predicted1))
        # bd1 = -(2/n1) * (y1 - y_predicted1)
        bd1 = -(2/n1) * sum(y1 - y_predicted1)
        # taking out the individual learning datapoints:
        m_curr1 = m_curr1 - learning_rate1 * md1
        b_curr1 = b_curr1 - learning_rate1 * bd1
        print("m {}, b {}, cost {}, iteration {}".format(m_curr1, b_curr1, cost1, i))

df = pd.read_csv('test_scores.csv')
x1 = df.math
y1 = df.cs

gradient_descent1(x1, y1)
