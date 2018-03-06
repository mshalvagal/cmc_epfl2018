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

