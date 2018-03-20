""" Lab 4 """
from exercise1 import exercise1
from exercise2 import exercise2
import biolog


def main():
    """Main function that runs all the exercises."""
    biolog.info('Implementing Lab 4 : Exercise 1')
    exercise1()
    biolog.info('Implementing Lab 4 : Exercise 2')
    exercise2()
    return


if __name__ == '__main__':
    from biopack import parse_args
    parse_args()
    main()

