"""This script introduces you to the usage of Functions in Python.
A function is defined in Python using the keyword def followed by function
name.
By default Python function returns none"""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # import biolog for log messages

### DEFINING FUNCTIONS ###

biolog.info(3*'\t' + 20*'#' + 'FUNCTIONS' + 20*'#' + 3*'\n')


# define a function with no arguments and no return values
def print_text():
    biolog.info('this is text')


# call the function
print_text()

biolog.info('A python file can contain more than one function!')


# define a function with one argument and no return values
def print_this(x):
    biolog.info(('Printing input \'x\'', x))


# call the function
print_this(3)       # prints 3
n = print_this(3)   # prints 3, but doesn't assign 3 to n
                    #   because the function has no return statement
biolog.warning(' prints 3, but doesn\'t assign 3 to n because the function has no return statement')


# define a function with one argument and one return value
def square_this(x):
    return x**2


# include an optional docstring to describe the effect of a function
def square_this(x):
    """Return the square of a number."""
    return x**2


# call the function
square_this(3)          # prints 9
var = square_this(3)    # assigns 9 to var, but does not print 9


# define a function with two 'positional arguments' (no default values) and
# one 'keyword argument' (has a default value)
def calc(a, b, op='add'):
    if op == 'add':
        return a + b
    elif op == 'sub':
        return a - b
    else:
        biolog.info('valid operations are add and sub')


biolog.info('Variables with default values should always be at the end!')
biolog.info('Variables can be directly assigned a value irrespective of their order if you use same name as in the function definition')


# call the function
calc(10, 4, op='add')   # returns 14
calc(10, 4, 'add')      # also returns 14: unnamed arguments are inferred by position
calc(10, 4)             # also returns 14: default for 'op' is 'add'
calc(10, 4, 'sub')      # returns 6
calc(10, 4, 'div')      # prints 'valid operations are add and sub'


# use 'pass' as a placeholder if you haven't written the function body
def stub():
    pass


# return two values from a single function
def min_max(nums):
    return min(nums), max(nums)


# return values can be assigned to a single variable as a tuple
nums = [1, 2, 3]
min_max_num = min_max(nums)         # min_max_num = (1, 3)

# return values can be assigned into multiple variables using tuple unpacking
min_num, max_num = min_max(nums)    # min_num = 1, max_num = 3
