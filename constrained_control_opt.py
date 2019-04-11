import numpy as np
from pyomo.environ import *

def one_dim_box_QP_opt(A, b, lower, upper):
    """
    Solve a box-constrained quadratic programming problem using Pyomo.
    Returns solution to the input constrained QP.
    """

    ipopt_executable = './ipopt'

    m = ConcreteModel()

    m.u = Var(domain=Reals)

    m.cost = Objective(expr=m.u*m.u*A + m.u*b, sense=minimize)

    m.box_lower = Constraint(expr=lower <= m.u)
    m.box_upper = Constraint(expr=m.u <= upper)

    SolverFactory('ipopt', executable=ipopt_executable).solve(m)

    temp_u = m.u()

    # Fix numerical precision issues
    if temp_u > upper:
        return upper

    if temp_u < lower:
        return lower

    return temp_u

