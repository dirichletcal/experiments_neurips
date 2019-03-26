#!/usr/bin/env python
from argparse import ArgumentParser

from calib.utils.summaries import load_all_csv
from calib.utils.summaries import create_summary_path
from calib.utils.summaries import generate_summaries
from calib.utils.summaries import generate_classifier_summaries
from calib.utils.summaries import generate_summary_hist


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

    print('Generating summaries of performance for uncalibrated classifiers')
    generate_classifier_summaries(df, summary_path, table_size='small')

    print('Generating summaries of performance and hyperparameters')
    generate_summaries(df, summary_path, table_size='small',
            hyperparameters=False, confusion_matrices=False)

    print('Generating Score histograms')
    df = load_all_csv(results_path, ".*score_histogram.csv")
    if 'Unnamed: 0' in df.columns:
        del df['Unnamed: 0']
    del df['filename']
    df.set_index(['dataset', 'classifier', 'calibration'], inplace=True)
    generate_summary_hist(df, summary_path)


if __name__ == '__main__':
    # __test_1()
    args = parse_arguments()
    main(**vars(args))
