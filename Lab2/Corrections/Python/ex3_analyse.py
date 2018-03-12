""" Analyse system symbolically (corrections) """

import numpy as np
import biolog
from ex3_pendulum import PendulumParameters, pendulum_system, pendulum_equation

USE_SYMPY = False
try:
    import sympy as sp
    USE_SYMPY = True
except ImportError as err:
    biolog.error(err)
    USE_SYMPY = False


def fixed_points_analysis():
    """ Exercise 3a - Compute and analyse fixed points """
    theta, g, L, d, time = sp.symbols("theta, g, L, d, t")
    params = PendulumParameters(g=g, L=L, d=d, sin=sp.sin)
    dt_sym = theta(time).diff(time)
    d2t_sym = theta(time).diff(time, 2)
    # System
    sys_matrix = sp.Matrix(pendulum_system(theta, dt_sym, parameters=params))
    sys = sp.Equality(
        sp.Matrix(np.array([[dt_sym], [d2t_sym]])),
        sys_matrix
    )
    biolog.info(u"Pendulum system:\n{}".format(sp.pretty(sys)))
    # Solve when dtheta = 0 and d2theta=0
    dt = 0
    d2t = 0
    t_sol = sp.solveset(
        sp.Equality(d2t, pendulum_equation(theta, dt, parameters=params)),
        theta  # Theta, the variable to be solved
    )
    sys_fixed = sp.Equality(
        sp.Matrix(np.array([[0], [0]])),
        sys_matrix
    )
    biolog.info(u"Solution for fixed points:\n{}\n{}\n{}\n{}".format(
        sp.pretty(sys_fixed),
        sp.pretty(sp.Equality(theta, t_sol)),
        sp.pretty(sp.Equality(dt_sym, dt)),
        sp.pretty(sp.Equality(d2t_sym, d2t))
    ))
    # Alternative way of solving
    sol = sp.solve(sys_fixed, [theta, dt_sym])
    biolog.info(u"Solution using an alternative way of solving:\n{}".format(
        sp.pretty(sol)
    ))
    # Jacobian, eigenvalues and eigenvectors
    jac = sys_matrix.jacobian(
        sp.Matrix([theta, dt_sym])
    )
    eigenvals = jac.eigenvals()
    eigenvects = jac.eigenvects()
    biolog.info(u"Jacobian:\n{}\nEigenvalues:\n{}\nEigenvectors:\n{}".format(
        sp.pretty(jac),
        sp.pretty(eigenvals),
        sp.pretty(eigenvects)
    ))
    # Stability analysis
    for i, s in enumerate(sol):
        biolog.info(u"Case {}:\n{}".format(
            i+1,
            sp.pretty(sp.Equality(sp.Matrix([theta, dt_sym]), sp.Matrix(s)))
        ))
        res = [None for _ in eigenvals.keys()]
        for j, e in enumerate(eigenvals.keys()):
            res[j] = e.subs({
                theta: s[0],
                dt_sym: s[1],
                g: 9.81,
                L: 1,
                d: 0.1
            })
            biolog.info(u"{}\n{}".format(
                sp.pretty(
                    sp.Equality(
                        e,
                        res[j]
                    )
                ),
                "{} {}".format(
                    sp.re(res[j]),
                    "> 0" if sp.re(res[j]) > 0 else "< 0"
                )
            ))
        fixed_point_type = None
        if all([sp.re(r) < 0 for r in res]):
            fixed_point_type = " stable point"
        elif all([sp.re(r) > 0 for r in res]):
            fixed_point_type = "n unstable point"
        else:
            fixed_point_type = " saddle point"
        biolog.info("Fixed point is a{}".format(fixed_point_type))
    return


def fixed_points():
    if USE_SYMPY:
        fixed_points_analysis()
    return