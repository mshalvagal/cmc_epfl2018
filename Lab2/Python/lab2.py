""" Lab 2 """


from exercise3 import exercise3
from exercise4 import exercise4


def main():
    """ Main """
    exercise3()
    exercise4()
    return


if __name__ == '__main__':
    from biopack import parse_args
    parse_args()
    main()

