import numpy as np
from scipy import optimize


# 数据
pi = [0, 0.01, 0.02, 4.5/100, 6.5/100]
ri = [0, 0.28, 0.21, 0.23, 0.25]
qi = [0, 2.5/100, 1.5/100, 5.5/100, 2.6/100]
ui = [0, 1.03, 1.98, 0.52, 0.40]

k = 0.1
x0_bound = (None, None)
x1_bound = x2_bound = x3_bound = x4_bound=(0, 1)
A_eq = [[0, 1.025, 1.015, 1.055, 1.026]]
B_eq = [1]
for i in range(0, 4):
    qi[i] = qi[i] - ri[i]
a = [qi, [-1, 2.5/100, 0, 0, 0], [-1, 0, 1.5/100 ,0, 0], [-1, 0, 0, 5.5/100, 0], [-1, 0, 0, 0, 2.6/100]]
b = [k, 0, 0, 0, 0]
z = [1, 0, 0, 0, 0]
res = optimize.linprog(z, A_ub=a, b_ub=b, bounds=(x0_bound, x1_bound, x2_bound, x3_bound, x4_bound), A_eq=A_eq, b_eq = B_eq)
print(res)


