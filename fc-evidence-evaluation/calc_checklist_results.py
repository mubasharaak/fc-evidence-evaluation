"""
Script to calcualte difference between base data and tests for atomic_reference_prec_recall scorer
"""
import os.path

import pandas as pd

import utils

_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/checklist"
_RESULTS_INPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results.csv")
_RESULTS_OUTPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results_diff.csv")


def main():
    # load results file
    results = pd.read_csv(_RESULTS_INPUT_FILE)
    # get base scores
    base_prec = \
    results.loc[(results['test'] == 'base_data') & (results['scorer'] == 'atomic_reference_prec')]['score'].values[0]
    base_recall = results.loc[(results['test'] == 'base_data') & (results['scorer'] == 'atomic_reference_recall')][
        'score'].values[0]
    results_diff = []

    for i, row in results.iterrows():
        if row['test'] == 'base_data':
            continue
        if "_prec" in row['scorer']:
            base = base_prec
        else:
            base = base_recall
        results_diff.append(
            {'test': row['test'],
             'scorer': row['scorer'],
             'diff': utils.percentage_difference(base, row['score'])})
    pd.DataFrame(results_diff).to_csv(_RESULTS_OUTPUT_FILE)


if __name__ == '__main__':
    main()
