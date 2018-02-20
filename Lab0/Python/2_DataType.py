"""This script explains the basic data types used in Python.
All the comman data types used in programming langauges are in Python too.
"""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # Import the biolog module to display log messages
biolog.info(3*'\t' + 20*'#' + 'DATA TYPES' + 20*'#' + 3*'\n')

biolog.warning(
    "In Python every element is treated as an Object."
    + " Including numbers and literals!!!"
)

# Different Data types
# Use 'type' method to identify the types
biolog.info('Data Type of 2 is : {}'.format(type(2)))  # int

biolog.info('Data Type of 2.0 is : {}'.format(type(2.0)))  # Float

biolog.info('Data Type of \'two\' is : {}'.format(type('two')))  # String

biolog.warning('There is no separate char data type in Python.')

biolog.info('Data Type of keyword True is : {}'.format(type(True)))  # Boolean

biolog.info('Data Type of keyword None is : {}'.format(type(None)))  # None type
