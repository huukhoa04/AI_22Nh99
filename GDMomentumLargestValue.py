import numpy as np
import matplotlib.pyplot as plt
import math

def cost(x):
    return (math.exp(-x) - 4 / math.exp(x)) ** 2
def grad(x):
    return -18*(math.exp(-2*x)) 

def gradient_descent(grad, start, learn_rate, n_iter, tolerance=1e-03):
        x = start
        for _ in range(n_iter):
            if abs(grad(x)) < tolerance:
                break
            x += learn_rate * grad_value
        return x

start = 0.0 # init x
learn_rate = 0.1 # learning rate
n_iter = 1000 # số lần lặp


x_max = gradient_descent(grad(start), start, learn_rate, n_iter)
print(f"The largest value of cost(x) is at x = {x_max}, cost(x) = {cost(x_max)}")