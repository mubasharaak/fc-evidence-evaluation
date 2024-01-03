import argparse

import properties
import reference_scorer

parser = argparse.ArgumentParser(
    description='Reference Scorer arguments'
)
parser.add_argument(
    '--training_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/fever_train_based.jsonl",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--dev_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/fever_dev_based.jsonl",
    help='Path to dev data for reference scorer'
)
parser.add_argument(
    '--test_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/fever_test_based.jsonl",
    help='Path to test data for evaluating fine-tuned reference scorer'
)
parser.add_argument(
    '--output_dir',
    default="./results/reference_scorer",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--results_filename',
    default="results.json",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--samples_filename',
    default="prediction_samples.txt",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--dataset',
    default="fever",
    choices=list(properties.Dataset),
    help='Dataset that is used for evaluation.'
)
parser.add_argument(
    '--hf_model',
    default="lucadiliello/BLEURT-20",
    help='Dataset that is used for evaluation.'
)

args = parser.parse_args()
_TRAIN_DATASET_PATH = args.training_data_path
_TEST_DATASET_PATH = args.dev_data_path
_DEV_DATASET_PATH = args.test_data_path

_OUTPUT_DIR = args.output_dir
_RESULTS_FILENAME = args.results_filename
_SAMPLES_FILENAME = args.samples_filename
_DATASET = args.dataset
_HG_MODEL_HUB_NAME = args.hf_model

_BATCH_SIZE = 32
_EPOCHS = 15


def main():
    reference_scorer.run_reference_scorer(_TRAIN_DATASET_PATH, _DEV_DATASET_PATH, _TEST_DATASET_PATH, _OUTPUT_DIR,
                                          _RESULTS_FILENAME, _SAMPLES_FILENAME, _HG_MODEL_HUB_NAME, train=True)


if __name__ == '__main__':
    main()
