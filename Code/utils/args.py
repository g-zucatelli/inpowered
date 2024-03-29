import argparse

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("-c",
                           "--config_filepath",
                           dest="config_filepath",
                           metavar="C",
                           default="None",
                           help="The Configuration File")
    
    args = argparser.parse_args()
    return args