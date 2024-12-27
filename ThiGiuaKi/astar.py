import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
# get data
data = genfromtxt('data.csv', delimiter=',')
n = data[0, 0]
C = data[:, 1]
D = data[:, 2]
