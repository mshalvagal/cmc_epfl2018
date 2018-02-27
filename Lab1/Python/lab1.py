""" Lab 1 """

from exercise1 import exercise1
from exercise2 import exercise2


def main():
    """ Main """
    exercise1()
    exercise2()
    return


if __name__ == "__main__":
    from biopack import parse_args
    parse_args()
    main()