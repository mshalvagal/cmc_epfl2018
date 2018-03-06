"""This script introduces you to the useage of Imports in Python.
One of the most powerful tool of any programming langauge is to be able to resuse code.
Python allows this by setting up modules. One can import existing libraries using the import function."""

### IMPORTS  ###

from __future__ import print_function  # Only necessary in Python 2
import biolog

biolog.info(3*'\t' + 20*'#' + 'IMPORTS' + 20*'#' + 3*'\n')

# A generic import of a default module named math
import math
# Now you have access to all the functionality availble
# in the math module to be used in this function
print('Square root of 25 computed from math module : {}'.format(math.sqrt(25)))

# To import a specific function from a module
from math import sqrt
# Now you can avoid referencing that the sqrt function is from
# math module and directly use it.
print('Square root of 25 computed from math module by importing only sqrt function: ', sqrt(25))

# Import a user defined module
# Here we import biolog : Module developed to display log messages for the exercise

biolog.info('Module developed to display log messages for the exercies')
biolog.warning("When you explicitly import functions from modules, it can lead to naming errors!!!""")

# Importing multiple functions from the same module
from math import sqrt, cos

# Defining an alias :
# Often having to reuse the actual name of module can be a pain.
# We can assign aliases to module names to avoid this problem
import datetime as dt
biolog.info("Here we import the module datetime as dt.")

# Getting to know the methods availble in a module
biolog.info(dir(math))
