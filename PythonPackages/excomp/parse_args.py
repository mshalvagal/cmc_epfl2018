""" Parse arguments """


def parse_script_args():
    """ Parse script arguments """
    import argparse
    parser = argparse.ArgumentParser(description=(
        "Parse exercise master file, then generate statement and corrections"
    ))
    parser.add_argument("file", type=str, help="Master file to compile")
    args = parser.parse_args()
    return args
