import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Simhei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False  # 让可以显示负号


def read_data(string):
    infile = open(string, 'r')
    data, l_x, l_y = [], [], []
    for line in infile:
        words = line.split(',')  # 以逗号分开
        x1 = float(words[0])
        x2 = float(words[1])
        y1 = int(words[2][0:1])
        l_x.append([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2,
                    x1 * x2 ** 2, x2 * x1 ** 2, x1 ** 3, x2 ** 3,
                    x1 * x2 ** 3, x1 ** 2 * x2 ** 2, x1 ** 3 * x2, x1 ** 4, x2 ** 4])
        l_y.append([y1])
        data.append([x1, x2, y1])
    infile.close()
    l_x = np.array(l_x)
    l_y = np.array(l_y)
    data = np.array(data)
    return data, l_x, l_y


def sigmoid(X):
    f = 1.0 / (1 + np.exp(-X))
    return f


def cost(theta, X, y):
    first = np.multiply((1 - y), (-np.dot(X, theta)))
    second = np.log(1 + np.exp(-np.dot(X, theta)))
    return np.sum(first - second)


def gradient_ascent(X, y, eta=0.1, n_iterations=5000):
    theta = np.random.randn(15, 1)
    for iteration in range(n_iterations):
        f = sigmoid(np.dot(X, theta))
        gradients = np.dot(X.T, (y - f))
        theta_old = np.array([i for i in theta])
        theta += eta * gradients
        if 0.01 < cost(theta_old, X, y) - cost(theta, X, y) < 0.01:
            return theta
        if 0.01 < np.dot((theta - theta_old).T, (theta - theta_old)) < 0.01:
            return theta
    return theta


def stochastic_gradient_ascent(X, y):
    theta = np.random.randn(15, 1)
    for epoch in range(100):
        for i in range(len(y)):
            random_index = np.random.randint(len(y))
            xi = X[random_index:random_index + 1, :]
            yi = y[random_index:random_index + 1]
            f = sigmoid(np.dot(xi, theta))
            gradients = np.dot(xi.T, (yi - f))
            eta = 1 / (np.sqrt(epoch + 1))  # 不断降低学习率
            theta += eta * gradients
    return theta


def mini_batch_gradient_ascent(X, y, mb_size=10, eta=0.1):
    theta = np.random.randn(15, 1)
    m = len(y)
    nums = (m - 1) // mb_size
    index_list = np.arange(nums)
    index_list = index_list * mb_size
    for epoch in range(100):
        for index in index_list:
            xi = X[index:index + mb_size + 1, :]
            yi = y[index:index + mb_size + 1]
            f = sigmoid(np.dot(xi, theta))
            gradients = np.dot(xi.T, (yi - f))
            theta += eta * gradients
        # 处理最后可能不到mb_size大小的数组
        d = m - nums * mb_size
        index = index_list[-1] + mb_size
        xi = X[index:index + d + 1, :]
        yi = y[index:index + d + 1]
        f = sigmoid(np.dot(xi, theta))
        gradients = np.dot(xi.T, (yi - f))
        theta += eta * gradients
    return theta


def reg_cost(theta, X, y, learning_rate=0.1):
    first = np.multiply((1 - y), (-np.dot(X, theta)))
    second = np.log(1 + np.exp(-np.dot(X, theta)))
    reg = (learning_rate / 2) * np.dot(theta.T, theta)
    return np.sum(first - second) + reg


def reg_gradient(X, y, eta=0.1, n_iterations=20000, learning_rate=0.1):
    theta = np.random.randn(15, 1)
    for iteration in range(n_iterations):
        f = sigmoid(np.dot(X, theta))
        gradients = np.dot(X.T, (y - f))
        theta_old = np.array([i for i in theta])
        theta = theta * (1 - eta * learning_rate) + eta * gradients
        if 0.01 < reg_cost(theta_old, X, y) - reg_cost(theta, X, y) < 0.01:
            return theta
        if 0.01 < np.dot((theta - theta_old).T, (theta - theta_old)) < 0.01:
            return theta
    return theta


def decision_function(x1, x2, theta):
    '''定义维度为3'''
    x = np.array([1, x1, x2,
                  x1 * x2, x1 ** 2, x2 ** 2,
                  x1 * x2 ** 2, x2 * x1 ** 2, x1 ** 3, x2 ** 3,
                  x1 * x2 ** 3, x1 ** 2 * x2 ** 2, x1 ** 3 * x2, x1 ** 4, x2 ** 4])
    return np.dot(x, theta)


def draw_scatter(data, title):
    label0 = data[:, 2] == 0
    label1 = data[:, 2] == 1
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.scatter(data[label1][:, 0], data[label1][:, 1], c='blue', label='label is 1')
    plt.scatter(data[label0][:, 0], data[label0][:, 1], c='red', marker='+', label='label is 0')
    plt.title(title)
    plt.legend()


def draw_decision_boundary(X, theta):
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    x1 = np.array(xx1).reshape(2500)
    x2 = np.array(xx2).reshape(2500)
    h = []
    for (i, j) in zip(x1, x2):
        h.append(decision_function(i, j, theta))
    h = np.array(h)
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, levels=[0], alpha=0.5, linestyles=['-'])


def predict(theta, X):
    p = sigmoid(np.dot(X, theta))
    p_label = []
    for i in p:
        p_label.append([1 if i >= 0.5 else 0])
    return p_label


def accuracy(X, y, theta):
    num = 1
    for (i, j) in zip(predict(theta, X), y):
        if i == j:
            num += 1
    accuracy = num / len(y) * 100
    return accuracy


def compare_rep():
    s = 'ex2data2.txt'
    data, X, y = read_data(s)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    theta = reg_gradient(X, y)
    accuracy1 = accuracy(X, y, theta)
    draw_scatter(data, u'惩罚因子为0.1时精度为%.4f' % accuracy1)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 2)
    theta = reg_gradient(X, y, learning_rate=0.01)
    accuracy1 = accuracy(X, y, theta)
    draw_scatter(data, u'惩罚因子为0.01时精度为%.4f' % accuracy1)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 3)
    theta = reg_gradient(X, y, learning_rate=0.2)
    accuracy1 = accuracy(X, y, theta)
    draw_scatter(data, u'惩罚因子为0.5时精度为%.4f' % accuracy1)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 4)
    theta = reg_gradient(X, y, learning_rate=0.5)
    accuracy1 = accuracy(X, y, theta)
    draw_scatter(data, u'惩罚因子为5时精度为%.4f' % accuracy1)
    draw_decision_boundary(X, theta)
    plt.show()


def main():
    s = 'ex2data2.txt'
    data, X, y = read_data(s)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    theta = gradient_ascent(X, y)
    print("梯度下降后的得到的θ值为：\n", theta.reshape(15))
    accuracy1 = accuracy(X, y, theta)
    draw_scatter(data, u'梯度下降, 精度为%.4f' % accuracy1)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 2)
    theta = stochastic_gradient_ascent(X, y)
    print("随机梯度下降后的得到的θ值为：\n", theta.reshape(15))
    accuracy2 = accuracy(X, y, theta)
    draw_scatter(data, u'随机梯度下降, 精度为%.4f' % accuracy2)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 3)
    theta = mini_batch_gradient_ascent(X, y, mb_size=3)
    print("小批量梯度下降后的得到的θ值为：\n", theta.reshape(15))
    accuracy4 = accuracy(X, y, theta)
    draw_scatter(data, u'小批量梯度下降, 精度为%.4f' % accuracy4)
    draw_decision_boundary(X, theta)
    plt.subplot(2, 2, 4)
    theta = reg_gradient(X, y)
    print("对梯度下降正则化后的得到的θ值为：\n", theta.reshape(15))
    accuracy3 = accuracy(X, y, theta)
    draw_scatter(data, u'正则化后梯度下降, 精度为%.4f' % accuracy3)
    draw_decision_boundary(X, theta)
    plt.show()


compare_rep()
