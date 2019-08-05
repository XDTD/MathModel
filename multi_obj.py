from scipy import optimize as op
import numpy as np
from gurobipy import *

try:

    # Create a new model
    m = Model("mip1")

    # Create variables
    x1 = m.addVar(vtype=GRB.BINARY, name="x1")
    x2 = m.addVar(vtype=GRB.BINARY, name="x2")
    x3 = m.addVar(vtype=GRB.BINARY, name="x3")
    x4 = m.addVar(vtype=GRB.BINARY, name="x4")
    x5 = m.addVar(vtype=GRB.BINARY, name="x5")
    x6 = m.addVar(vtype=GRB.BINARY, name="x6")
    x7 = m.addVar(vtype=GRB.BINARY, name="x7")


    # Set objective
    m.setObjective(x1 + x2 + x3 + x4 + x5 +x6 +x7, GRB.MINIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x1 >= 20, "c0")
    m.addConstr(x1+x2 >= 25,"c1")
    m.addConstr(x1+x2+x3 >= 10,"c2")
    m.addConstr(x1+x2+x3+x4 >= 30,"c3")
    m.addConstr(x2+x3+x4+x5 >= 20,"c4")
    m.addConstr(x3+x4+x5+x6 >= 10,"c5")
    m.addConstr(x7+x4+x5+x6 >= 5,"c6")
    m.addConstr(x2 >= 0, "c7")
    m.addConstr(x3 >= 0,"c8")
    m.addConstr(x4 >= 0,"c9")
    m.addConstr(x5 >= 0, "c10")
    m.addConstr(x6 >= 0,"c11")
    m.addConstr(x7 >= 0,"c12")



    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')



def Q_work():
    c = np.array([1, 1 , 1, 1])
    # A_ub = np.array([[-1, 0, 0, 0], [-1, -1, 0, 0], [-1, -1, -1, 0], [-1, -1, -1, -1], [0, -1, -1, -1], [0, 0, -1, -1], [0, 0, 0, -1]])
    # B_ub = np.array([-20, -25, -30, -10, -20, -10, -5])
    # # A_eq=np.array([[1,1,1]])
    # # B_eq=np.array([7])
    # x1 = (0, None)
    # x2 = (0, None)
    # x3 = (0, None)
    # x4 = (0,None)
    # res = op.linprog(c, A_ub, B_ub, bounds=(x1, x2, x3, x4))
    # print(res)



if __name__ == '__main__':
    Q_work()