"""This script discussess the basic math operations used in Python."""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # Import biolog module for log messages

biolog.info(3*'\t' + 20*'#' + 'MATH' + 20*'#' + 3*'\n')

# Basic operations
biolog.info("Adding 132 + 123 : {}".format((132 + 123)))  # Add

biolog.info("Subrtacting 43 - 23 : {}".format((43 - 23)))  # Subtract

biolog.info("Multiply 65 * 87 : {}".format((65 * 87)))  # Multiply

biolog.info("Exponent 65**4 : {}".format((65**4)))  # Exponent

biolog.info("Modulo 5 % 4 : {}".format((5 % 4)))  # Modulo

biolog.info("Division 10 / 4 : {}".format((10/4)))  # Division

biolog.warning("Returns 2 in Python 2, returns 2.5 in Python 3")

biolog.info("Division 10.0 / 4 : {}".format((10.0/4)))  # True division

biolog.info(
    "Force True Division in Python 2 by importing division from __future__"
)
