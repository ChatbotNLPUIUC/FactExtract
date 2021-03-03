import os
import argparse
from extract_independent import split_complete_sents
import nltk
import warnings

warnings.filterwarnings("ignore")

# Argparse uses the ArgumentTypeError to give a rejection message like:
# error: argument input: x does not exist
def extant_file(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError(f"File {x} does not exist")
    if not os.path.isfile(x):
        raise argparse.ArgumentTypeError(f"{x} is a directory, not a file")
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', required=True, type=extant_file)
    parser.add_argument('--dest_file', required=True, type=extant_file)
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":
    # get the args from command line
    opts = parse_args()
    # read input from the file
    with open(opts.source_file, 'r') as f:
        # makes sure to remove whitespace and empty srings
        all_text = [x.strip() for x in f.readlines() if x.strip()]
    # join them together into one string
    all_text = " ".join(all_text)
    # tokenize into list of sentences
    all_text = nltk.sent_tokenize(all_text)
    all_text = " ".join(split_complete_sents(all_text))
    print(all_text)