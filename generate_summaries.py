#!/usr/bin/env python
from argparse import ArgumentParser

from calib.utils.summaries import load_all_csv
from calib.utils.summaries import create_summary_path
from calib.utils.summaries import generate_summaries


def parse_arguments():
    parser = ArgumentParser(description=("Generates a summary of all the " +
                                         "experiments in the subfolders of " +
                                         "the specified path"))
    parser.add_argument("results_path", metavar='PATH', type=str,
                        default='results',
                        help="Path with the result folders to summarize.")
    parser.add_argument("summary_path", metavar='SUMMARY', type=str,
                        default=None, nargs='?',
                        help="Path to store the summary.")
    return parser.parse_args()


def main(results_path, summary_path):
    df = load_all_csv(results_path, ".*raw_results.csv")
    summary_path = create_summary_path(summary_path, results_path)
    generate_summaries(df, summary_path)


if __name__ == '__main__':
    # __test_1()
    args = parse_arguments()
    main(**vars(args))
