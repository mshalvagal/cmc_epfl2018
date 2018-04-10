""" Lab 5 """

import biolog
from exercise3 import exercise3
from exercise4 import exercise4


def main():
    """Main function that runs all the exercises."""
    biolog.info('Implementing Lab 5 : Exercise 3')
    exercise3()
    biolog.info('Implementing Lab 5 : Exercise 4')
    exercise4()
    return


if __name__ == '__main__':
    from biopack import parse_args
    parse_args()
    main()

