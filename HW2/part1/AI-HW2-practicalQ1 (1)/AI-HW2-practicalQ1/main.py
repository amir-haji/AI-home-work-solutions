import matplotlib.pyplot as plt
import numpy as np
import math
import random
from func import draw_points


def plot_func(func, start, end):
    x = []
    y = []

    point = start
    while point <= end:
        x.append(point)
        y.append(func(point))
        point += .01

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('function')
    plt.show()


def f1(x):
    ans = ((x**4)*(math.e ** x) - math.sin(x)) / 2
    return ans


def f2(x):

    ans = 5 * math.log(math.sin(5 * x) + math.sqrt(x), 10)
    return ans


def f3(x):
    ans = math.cos(5*math.log(x, 10)) - x ** 3/10
    return ans


def f4(x, y):
    ans = (2 ** x / 10000) + (math.e ** y / 20000) + (x ** 2) + (4 * y ** 2) - (2*x) - (3*y)
    return ans


def first_order_derivative(func, x):
    ans = (func(x + .00001) - func(x)) / .00001
    return ans


def second_derivative(func, x):
    ans = (first_order_derivative(func, x + .00001) - first_order_derivative(func, x)) / .00001
    return ans


def gradient(func, x, y):
    partial_x = (func(x + .00001, y) - func(x, y)) / .00001
    partial_y = (func(x, y + .00001) - func(x, y)) / .00001

    return [partial_x, partial_y]


def gradient_descent_1d(func, start, end, learning_rate, max_iter):
    x_point = random.uniform(start, end)

    while max_iter:
        derivative = first_order_derivative(func, x_point)

        x_point -= learning_rate * derivative

        max_iter -= 1

    return x_point


def newton_raphson(func, start, end, max_iter):
    x_point = random.uniform(start, end)

    while max_iter:
        f_derivative = first_order_derivative(func, x_point)
        s_derivative = second_derivative(func, x_point)

        x_point -= (f_derivative / s_derivative)
        max_iter -= 1

    return x_point


def gradient_descent_2d(func, start_x, end_x, start_y, end_y, learning_rate, max_iter):
    x_point = random.uniform(start_x, end_x)
    y_point = random.uniform(start_y, end_y)

    x_ans = [x_point]
    y_ans = [y_point]

    while max_iter:
        grad = gradient(func, x_point, y_point)
        x_point -= learning_rate * grad[0]
        y_point -= learning_rate * grad[1]

        x_ans.append(x_point)
        y_ans.append(y_point)
        max_iter -= 1

    return x_ans, y_ans


def simulated_annealing(func, start_x, end_x, initial_t, stopping_t, max_iter, gamma, alpha):
    current = random.uniform(start_x, end_x)
    t = initial_t
    while t > stopping_t and max_iter:
        t *= gamma
        new = random.uniform(current - alpha, current + alpha)
        delta = -func(new) + func(current)
        if delta > 0:
            current = new
        else:
            p = math.e ** (delta / t)
            number = random.uniform(0, 1)
            if number <= p:
                current = new

        max_iter -= 1

    return current




            
        
