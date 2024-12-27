import numpy as np
import matplotlib.pyplot as plt
import math

def cost(x):
    return (math.exp(-x) - 4 / math.exp(x)) ** 2
def grad(x):
    return -18*(math.exp(-2*x)) 

# check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new))/len(theta_new) < 0.001
    
# gd momentum
def GD_momentum(theta_init, alpha=0.1, beta=0.9):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(1000):
        v_new = beta*v_old + alpha*grad(theta[-1])
        theta_new = theta[-1] - v_new
        theta.append(theta_new)
        v_old = v_new
    return (theta,it)
# gd thường 
def myGD1(x0,alpha=0.1, gra = 1e-3, loop = 1000):
    x = [x0]
    for it in range(loop):
        x_new = x[-1] - alpha*grad(x[-1])
        if abs(grad(x_new)) < gra:
            break
        x.append(x_new)
    return (x, it)

if __name__ == '__main__':
    print("OK");
    (x3, it3) = myGD1(5,0.1)
    print('GD_Solution x3 = %f, cost = %f, obtained after %d iterations'%(x3[-1], cost(x3[-1]), it3))
    (x1, it1) = GD_momentum(5, 0.1, beta = 0.9)
    print('Momentum_Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))


