"""This script introduces you to the usage of Loops in Python.
Loops are useful to repeatedly to do a task over and over again.
Here we look at for and while loops in particular"""

from __future__ import print_function  # Only necessary in Python 2

import biolog  # import biolog for log messages

### FOR LOOPS AND WHILE LOOPS ###

biolog.info(3*'\t' + 20*'#' + 'FOR AND WHILE LOOPS' + 20*'#' + 3*'\n')

# range returns a list of integers (Python 2) or a sequence (Python 3)
# returns [0, 1, 2]: includes start value but excludes stop value
biolog.info('Using range method between 0 and 3 {}'.format(list(range(0, 3))))

biolog.info('A very useful method for iteration')

biolog.warning('Includes start value but excludes the stop values')

biolog.info('Python 2 returns a list and Python 3 returns a sequence')

list(range(3))        # equivalent: default start value is 0
list(range(0, 5, 2))  # returns [0, 2, 4]: third argument is the step value

# Python 2 only: use xrange to create a sequence rather than a list (saves memory)
range(100, 100000, 5)

# for loop (not the recommended style)
fruits = ['apple', 'banana', 'cherry']
biolog.warning('Not a Recommended style')
for i in range(len(fruits)):
    biolog.info((fruits[i].upper()))

# for loop (recommended style)
biolog.warning('Recommended style')
for fruit in fruits:
    biolog.info((fruit.upper()))

# iterate through two things at once (using tuple unpacking)
family = {'dad': 'homer', 'mom': 'marge', 'size': 6}
biolog.info('Iterating over two things at once :')
for key, value in list(family.items()):
    biolog.info((key, value))

# use enumerate if you need to access the index value within the loop
biolog.info('Indexing the list')
for index, fruit in enumerate(fruits):
    biolog.info((index, fruit))

# for/else loop
for fruit in fruits:
    if fruit == 'banana':
        biolog.info('Found the banana!')
        break    # exit the loop and skip the 'else' block
else:
    # this block executes ONLY if the for loop completes without hitting 'break'
    biolog.info("Can't find the banana")

# while loop
count = 0
while count < 5:
    biolog.info('This will print 5 times')
    count += 1    # equivalent to 'count = count + 1'
