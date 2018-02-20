"""This script introduces you to the usage of Numpy module in Python.
It is the most important scientific module in Python.
We will only go over some of the very basic functionalities of numpy here.
It can do a lot more!!!"""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # import biolog for log messages
import numpy as np

### NUMPY ###

biolog.info(3*'\t' + 20*'#' + 'NUMPY' + 20*'#')

biolog.info('It is a common standard to use the Numpy module as np!')

## Numpy is the fundamental scientific computing package in Python
## N-dimensional array object

# Create an array using np

A = np.array([[1,2,3],[4,5,6]])

biolog.info('Array created using numpy, A : {}'.format(A))

biolog.info('Type of numpy array A : {}'.format(type(A)))

biolog.info('You can also explicity mention the type of data in the array')

A = np.array([[1, 2, 3], [4, 5, 6]], np.float)

# Basic numpy methods similar to the ones in MATLAB
# arange, linspace, zeros, shape

biolog.info('Array of integers between 0 and 10 using arange method in numpy : \n{}'.format(np.arange(0,10,1)))

biolog.info('Array of integers between 0 and 10 using linspace  method in numpy : \n{}'.format(np.linspace(0,10,11)))

biolog.info('Array of zeros using zeros method in numpy :\n{}'.format(np.zeros((2, 3))))

biolog.info('Shape of an array using shape method in numpy :\n{}'.format(np.zeros((2,3)).shape))

biolog.warning('Numpy arrays are not matrices!!!')

# Math operations
## As mentioned numpy array's are not matrices
## Math operations operate element-wise by default

a = np.arange(4)

biolog.info('Array a : \n{}'.format(a))

b = np.array([2, 3, 2, 4])

biolog.info('Array b : \n{}'.format(b))

biolog.info('Element wise multiplication of numpy arrays a*b : \n{}'.format(a*b))

biolog.info('Element wise subtraction of numpy arrays b-a :\n{}'.format(b-a))

biolog.info('Have a look at Array broadcasting to understand how numpy compares two arrays\n for math operations')

# Create a random numpy array using random method
A = np.ones((2,3))
biolog.info('Numpy array  A of size (2,3) using ones method :\n{}'.format(A))

# Create a random numpy array using random method
B = np.ones((3,2))
biolog.info('Numpy array B of size (3,2) using ones method :\n{}'.format(B))

biolog.info('Matrix multiplication A*B using numpy dot method :\n{}'.format(np.dot(A, B)))

biolog.info('Matrix multiplication B*A using numpy dot method :\n{}'.format(np.dot(B, A)))

try:
    biolog.info('Matrix multiplication transpose(A)*B using numpy dot method :\n{}'.format(np.dot(A.T, B)))
except ValueError:
    biolog.error('ValueError: shapes (3,2) and (3,2) not aligned: 2 (dim 1) != 3 (dim 0)')

biolog.info('Have a look at numpy matrices method!')
biolog.warning(' Numpy matrices are not very often used!')
