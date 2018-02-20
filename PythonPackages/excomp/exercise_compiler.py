""" Exercise compiler to generate corrections from statement separately

The idea is to write one master file containing a few annotations to generate
the statement and the corrections for the CMC exercises. This allows to write
one working master file that generates both the statement and the corrections,
thus avoiding duplicate work and insuring the statement and the corrections
are always up-to-date with each other.

Furthermore, the developer can precise if a line or block of lines should be
commented or not in the destined output.

Guidelines for the master file:
===============================

- Should contain as few annotations as possible
- Should be able to run the same code as the corrections file
- All the lines featured in the statement file should be present, but commented

# Instructions:
===============

Types {type} are either "S" for statement file and "C" for corrections file.
The standard metasymbol is written as follows: # _{type}

In order to only feature a line in the {type} output:

line_where_code_should_only_appear_in_certain_file  # _{type} (# _c)

In order to only feature a block of lines in the {type} output:

# _{type}0 (# _c)
block
of
lines
# _{type}1

Additional notes:
=================

- The # _c symbol is optional and allows to establis if the line(s) shoudl be
commented or not in the final output.
- The # S and # C character are removed in final files

Expected outputs:
=================

NO INPUT

    >>> compile_string(
    ...     "",
    ...     "# _C", "# _S"
    ... )
    ''

    >>> compile_string(
    ...     "",
    ...     "# _S", "# _C"
    ... )
    ''

BASIC

    >>> compile_string(
    ...     "    This\\n    is a  # _S\\n    test",
    ...     "# _C", "# _S"
    ... )
    '    This\\n    test'

SINGLE LINE ANNOTATIONS:

    STATEMENT

        >>> compile_string(
        ...     "line_of_code  # _S",
        ...     "# _S", "# _C"
        ... )
        'line_of_code'

        >>> compile_string(
        ...     "line_of_code  # _C",
        ...     "# _S", "# _C"
        ... )
        ''

        >>> compile_string(
        ...     "line_of_code  # _S  # _c",
        ...     "# _S", "# _C"
        ... )
        '# line_of_code'

        >>> compile_string(
        ...     "line_of_code  # _C # _c",
        ...     "# _S", "# _C"
        ... )
        ''

    CORRECTIONS

        >>> compile_string(
        ...     "line_of_code  # _C",
        ...     "# _C", "# _S"
        ... )
        'line_of_code'

        >>> compile_string(
        ...     "line_of_code  # _C  # _c",
        ...     "# _C", "# _S"
        ... )
        '# line_of_code'

        >>> compile_string(
        ...     "line_of_code  # _C  # _c",
        ...     "# _C", "# _S"
        ... )
        '# line_of_code'

        >>> compile_string(
        ...     "line_of_code  # _S # _c",
        ...     "# _C", "# _S"
        ... )
        ''

BLOCK ANNOTATIONS:

    STATEMENT

        >>> compile_string(
        ...     "# _S0\\nline\\nof\\ncode\\n# _S1",
        ...     "# _S", "# _C"
        ... )
        'line\\nof\\ncode'

        >>> compile_string(
        ...     "# _S0 # _c\\nline\\nof\\ncode\\n# _S1",
        ...     "# _S", "# _C"
        ... )
        '# line\\n# of\\n# code'

        >>> compile_string(
        ...     "# _S0 # _u\\n# line\\n# of\\n# code\\n# _S1",
        ...     "# _S", "# _C"
        ... )
        'line\\nof\\ncode'

        >>> compile_string(
        ...     "% _S0 % _u\\n% line\\n% of\\n% code\\n% _S1",
        ...     "% _S", "% _C"
        ... )
        'line\\nof\\ncode'

        >>> compile_string(
        ...     "# _C0\\nline\\nof\\ncode\\n# _C1",
        ...     "# _S", "# _C"
        ... )
        ''

    CORRECTIONS

        >>> compile_string(
        ...     "# _C0\\nline\\nof\\ncode\\n# _C1",
        ...     "# _C", "# _S"
        ... )
        'line\\nof\\ncode'

        >>> compile_string(
        ...     "# _C0 # _c\\nline\\nof\\ncode\\n# _C1",
        ...     "# _C", "# _S"
        ... )
        '# line\\n# of\\n# code'

        >>> compile_string(
        ...     "# _C0 # _u\\n# line\\n# of\\n# code\\n# _C1",
        ...     "# _C", "# _S"
        ... )
        'line\\nof\\ncode'

        >>> compile_string(
        ...     "# _S0\\nline\\nof\\ncode\\n# _S1",
        ...     "# _C", "# _S"
        ... )
        ''

"""

import os
import biolog

SYM_FLAG = "#"
SYM_META = "{} _{}".format(SYM_FLAG, "{}")
SYM_STAT = SYM_META.format("S")
SYM_CORR = SYM_META.format("C")
SYM_COMM = SYM_META.format("c")
SYM_UCOM = SYM_META.format("u")


def opts(**kwargs):
    """ Modify options """
    global SYM_FLAG
    global SYM_META
    global SYM_STAT
    global SYM_CORR
    global SYM_COMM
    global SYM_UCOM
    SYM_FLAG = kwargs.pop("SYM_FLAG", SYM_FLAG)
    SYM_META = kwargs.pop("SYM_META", SYM_META)
    SYM_STAT = kwargs.pop("SYM_STAT", SYM_STAT)
    SYM_CORR = kwargs.pop("SYM_CORR", SYM_CORR)
    SYM_COMM = kwargs.pop("SYM_COMM", SYM_COMM)
    SYM_UCOM = kwargs.pop("SYM_UCOM", SYM_UCOM)
    return


def check_and_create_dir(filename):
    """ Check if directory exists and create it if not """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return


def clean_end_spaces(text):
    """ Remove spaces at the end of lines

    >>> clean_end_spaces("a \\n  \\n    b \\n")
    'a\\n\\n    b\\n'
    """
    while " \n" in text:
        text = text.replace(" \n", "\n")
    while text and text[-1] == " ":
        text = text[:-1]
    return text


def parse_add_remove_block(text, symbol_insert, symbol_remove):
    """ Parse lines to leave and remove

    >>> parse_add_remove_block("start\\nb0\\naaa\\nb1\\nend", "b", "c")
    'start\\naaa\\nend'
    >>> parse_add_remove_block("start\\nc0\\naaa\\nc1\\nend", "b", "c")
    'start\\nend'
    """
    symbol_insert_b = [symbol_insert + "{}".format(i) for i in range(2)]
    symbol_remove_b = [symbol_remove + "{}".format(i) for i in range(2)]
    symbol_list = (
        [symbol_insert_b[i] for i in range(2)]
        + [symbol_remove_b[i] for i in range(2)]
    )
    text = parse_remove_block(text, symbol_remove_b)
    text = parse_add_block(text, symbol_insert_b)
    if any([symbol in text for symbol in symbol_list]):
        raise Exception("Some beginning or end of blocks are orphins")
    text = clean_end_spaces(text)
    return text


def parse_remove_block(text, symbol_remove_b):
    """ Remove block """
    while symbol_remove_b[0] in text:
        lines = text.splitlines()
        beg_lines = [i for i, l in enumerate(lines) if symbol_remove_b[0] in l]
        end_lines = [i for i, l in enumerate(lines) if symbol_remove_b[1] in l]
        # Find beginning and end of section
        if end_lines[0] < beg_lines[0]:
            raise Exception("Block end found before block start")
        lines = lines[:beg_lines[0]] + lines[end_lines[0]+1:]
        text = "\n".join(lines)
    text = clean_end_spaces(text)
    return text


def parse_add_block(text, symbol_insert_b):
    """ Add block

    >>> parse_add_block(
    ...     "    start\\n    # x0 # _c\\n    aaa\\n    # x1\\n    end",
    ...     ["x0", "x1"]
    ... )
    '    start\\n    # aaa\\n    end'
    >>> parse_add_block(
    ...     "    start\\n    # x0 # _c\\n    # aaa\\n    # x1\\n    end",
    ...     ["x0", "x1"]
    ... )
    '    start\\n    # # aaa\\n    end'
    >>> parse_add_block(
    ...     "    start\\n    # x0 # _u\\n    aaa\\n    # x1\\n    end",
    ...     ["x0", "x1"]
    ... )
    '    start\\n    aaa\\n    end'
    >>> parse_add_block(
    ...     "    start\\n    # x0 # _u\\n    # aaa\\n    # x1\\n    end",
    ...     ["x0", "x1"]
    ... )
    '    start\\n    aaa\\n    end'
    """
    while symbol_insert_b[0] in text and text:
        # Split into lines
        lines = text.splitlines()
        beg_lines = [i for i, l in enumerate(lines) if symbol_insert_b[0] in l]
        end_lines = [i for i, l in enumerate(lines) if symbol_insert_b[1] in l]
        # Find beginning and end of section
        if end_lines[0] < beg_lines[0]:
            raise Exception("Block end found before block start")
        # Find lines
        flag_c, flag_u = False, False
        if SYM_COMM in lines[beg_lines[0]]:
            flag_c = True
        elif SYM_UCOM in lines[beg_lines[0]]:
            flag_u = True
        if flag_c or flag_u:  # Find tab
            line = lines[beg_lines[0]]
            tab = " "*(len(line) - len(line.lstrip(' ')))
        sublines = lines[beg_lines[0]+1:end_lines[0]]
        # Comment
        if flag_c:
            sublines = [sl.replace(tab, tab + SYM_FLAG+" ") for sl in sublines]
        elif flag_u:
            sublines = [sl.replace(SYM_FLAG+" ", "") for sl in sublines]
        lines[beg_lines[0]+1:end_lines[0]] = sublines
        # Remove beginning and end of section annotations
        lines.pop(end_lines[0])
        lines.pop(beg_lines[0])
        text = "\n".join(lines)
    text = clean_end_spaces(text)
    return text


def parse_add_remove_line(text, symbol_insert, symbol_remove):
    """ Parse lines to leave and remove

    >>> parse_add_remove_line("a  ", "b", "c")
    'a'
    >>> parse_add_remove_line("ab  ", "b", "c")
    'a'
    >>> parse_add_remove_line("ac  ", "b", "c")
    ''
    >>> parse_add_remove_line("abc  ", "b", "c")
    Traceback (most recent call last):
    ...
    Exception: Contradictory symbols at line 0
    """
    text = text.splitlines()
    for i, line in enumerate(text):
        if symbol_insert in line and symbol_remove in line:
            raise Exception("Contradictory symbols at line {}".format(i))
        elif symbol_insert in line:
            text[i] = line.replace(symbol_insert, "")
        elif symbol_remove in line:
            text[i] = "REMOVELINE"
    text = [line for line in text if line != "REMOVELINE"]
    text = "\n".join(text)
    text = clean_end_spaces(text)
    return text


def parse_add_remove(text, symbol_insert, symbol_remove):
    """ Parse lines to leave and remove """
    text = parse_add_remove_block(text, symbol_insert, symbol_remove)
    text = parse_add_remove_line(text, symbol_insert, symbol_remove)
    return text


def parse_comments(text):
    """ Parse comments

    >>> parse_comments("hello {}".format(SYM_COMM))
    '# hello'
    >>> parse_comments("# hello {}".format(SYM_COMM))
    '# hello'
    >>> parse_comments("# hello {}".format(SYM_UCOM))
    'hello'
    >>> parse_comments("hello {}".format(SYM_UCOM))
    'hello'
    """
    text = text.splitlines()
    for i, line in enumerate(text):
        if SYM_COMM in line and SYM_UCOM in line:
            raise Exception(
                "Both comment and uncomment found at line {}".format(i)
            )
        elif SYM_COMM in line:
            spaces = 0
            line = line.replace(SYM_COMM, "")
            for spaces, c in enumerate(line):
                if c != " ":
                    break
            text[i] = (
                line[:spaces] + SYM_FLAG+" " + line[spaces:]
                if line[spaces] != SYM_FLAG  # If not already a comment
                else line
            )
        elif SYM_UCOM in line:
            spaces = 0
            line = line.replace(SYM_UCOM, "")
            for spaces, c in enumerate(line):
                if c != " ":
                    break
            text[i] = line.replace(" "*spaces+SYM_FLAG+" ", " "*spaces, 1)
    return clean_end_spaces("\n".join(text))


def compile_string(text, symbol_insert, symbol_remove):
    """ Compile string """
    text = clean_end_spaces(text)
    text = parse_add_remove(text, symbol_insert, symbol_remove)
    text = clean_end_spaces(text)
    text = parse_comments(text)
    text = clean_end_spaces(text)
    return text


def _compile(master, filename, symbol_insert, symbol_remove):
    """ Parse master accrding to symbols """
    text = open(master, "r").read()
    text = compile_string(text, symbol_insert, symbol_remove)
    file_text = open(filename, "w+")
    file_text.write(text)
    return


def exercise_compile(master_file, **kwargs):
    """ Compile exercise master file

    Kwargs:
    - folder_s: Statement output folder
    - folder_c: Corrections output folder
    """
    master_name, master_extension = master_file.split(".")
    statement_file = "{}/{}{}.{}".format(
        kwargs.pop("folder_s", "Statement/Python"),
        master_name,
        kwargs.pop("end_s", ""),
        master_extension
    )
    corrections_file = "{}/{}{}.{}".format(
        kwargs.pop("folder_c", "Corrections/Python"),
        master_name,
        kwargs.pop("end_c", ""),
        master_extension
    )
    for filename, symbol_insert, symbol_remove in [
            [statement_file, SYM_STAT, SYM_CORR],
            [corrections_file, SYM_CORR, SYM_STAT]
    ]:
        biolog.info("Compiling {}".format(filename))
        check_and_create_dir(filename)
        _compile(master_file, filename, symbol_insert, symbol_remove)
    return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
