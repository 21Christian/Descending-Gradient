import numpy as np
import pandas as pd

# loading the data
df = pd.read_csv('test_scores.csv')
math_values = []
cs_values = []

# filling the lists with the data from the file and creating arrays
for i in df.math.values:
    math_values.append(i)
for j in df.cs.values:
    cs_values.append(j)

x = np.asarray(math_values)
y = np.asarray(cs_values)

def gradient_descent(x, y):
    m, b = 0, 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    for k in range(iterations):
        y_predicted = m * x + b
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum((y - y_predicted))
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m, b, cost, k))



gradient_descent(x, y)