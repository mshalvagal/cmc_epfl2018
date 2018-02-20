""" Package for compiling CMC course exercises """

from .exercise_compiler import exercise_compile, opts


if __name__ == '__main__':
    from parse_args import parse_script_args
    SCRIPT_ARGS = parse_script_args()
    opts()
    exercise_compile(SCRIPT_ARGS.file)
