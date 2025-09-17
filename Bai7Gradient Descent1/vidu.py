from __future__ import division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 2 * x + 5 * np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)
def MyGD1(eta,x0) :
    x=[x0]
    for it in range(100):
        x_new = x[-1]-eta*grad(x[-1])
        if abs(grad(x_new))<1e-3:
            break
        x.append(x_new)
    return (x,it)
(x1, it1) = MyGD1(.1, -5)
(x2, it2) = MyGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))

# x_vals = np.linspace(-6, 6, 100)
# y_vals = cost(x_vals)
# plt.plot(x_vals, y_vals)
# plt.scatter([x1[-1], x2[-1]], [cost(x1[-1]), cost(x2[-1])], color='red')  # vẽ điểm cực tiểu
# plt.title('Cost function')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.grid()
# plt.show()

