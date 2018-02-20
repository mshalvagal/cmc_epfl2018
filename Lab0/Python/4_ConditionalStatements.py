"""This script discussess the different conditional statements in Python."""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # Import biolog module for log messages

biolog.info(3*'\t' + 20*'#' + 'CONDITIONAL STATEMENTS' + 20*'#' + 3*'\n')


# Before dwelling into the conditional statements, first let us look at the
# Comparison and Boolean Operators availble

# Comparison and Boolean Operations

x = 5  # Assignment Statement
# Remember that Python treats every element in the program as an Object

# Comparisons
biolog.info("x > 3 : {}".format((x > 3)))  # Greater than

biolog.info("x < 3 : {}".format((x < 3)))  # Lesser than

biolog.info("x >= 3 : {}".format((x >= 3)))  # Greater than or equal to

biolog.info("x != 3 : {}".format((x != 3)))  # Not equal to

biolog.info("x == 3 : {}".format((x == 3)))  # Equal to

# Boolean operations
biolog.info("(5 > 3) and (6 < 3) : {}".format((5 > 3)and(6 < 3)))  # and operator

biolog.info("(5 > 3) or (6 > 3) : {}".format((5 > 3)or(6 > 3)))  # or operator

biolog.info("not False : {}".format((not False)))  # not operator

# Evaluation order : not, and, or
biolog.info("False or not False and True : {}".format(False or not False and True))

biolog.info("Try to implement >= operator with \'or\' operator")

# Conditional Statements

# if statement
if x > 0:
    biolog.info('Positive')

# if/else statement
if x > 0:
    biolog.info('Positive')
else:
    biolog.info('Zero or Negative')

# if/elif/else statement
if x > 0:
    biolog.info('Positive')
elif x == 0:
    biolog.info('Zero')
else:
    biolog.info('Negative')

# Ternary operator
check = 'Positive' if x > 0 else 'Zero or Negative'
biolog.info('Ternary operator : {}'.format(check))
