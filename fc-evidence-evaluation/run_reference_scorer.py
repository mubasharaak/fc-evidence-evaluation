import argparse

import properties
import reference_scorer

parser = argparse.ArgumentParser(
    description='Reference Scorer arguments'
)
parser.add_argument(
    '--training_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/bleurt_finetune_train.jsonl",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--dev_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/bleurt_finetune_dev.jsonl",
    help='Path to dev data for reference scorer'
)
parser.add_argument(
    '--test_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data/bleurt_finetune_test.jsonl",
    help='Path to test data for evaluating fine-tuned reference scorer'
)
parser.add_argument(
    '--output_dir',
    default="/scratch/users/k20116188/fc_evidence_evaluation/results/reference_scorer",
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
    '--hf_model',
    default="Elron/bleurt-base-512",
    help='Path to HG bleurt model.'
)
parser.add_argument(
    '--train',
    default=True,
    action="store_false",
    help='If set, fine-tunes scorer with data specified through --training_data_path'
)


args = parser.parse_args()
_TRAIN_DATASET_PATH = args.training_data_path
_TEST_DATASET_PATH = args.dev_data_path
_DEV_DATASET_PATH = args.test_data_path

_OUTPUT_DIR = args.output_dir
_RESULTS_FILENAME = args.results_filename
_SAMPLES_FILENAME = args.samples_filename
_HG_MODEL_HUB_NAME = args.hf_model

_TRAIN = args.train

_BATCH_SIZE = 4
_EPOCHS = 5


def main():
    reference_scorer.run_reference_scorer(train_dataset_path=_TRAIN_DATASET_PATH, dev_dataset_path=_DEV_DATASET_PATH,
                                          test_dataset_path=_TEST_DATASET_PATH, output_path=_OUTPUT_DIR,
                                          results_filename=_RESULTS_FILENAME, samples_filenames=_SAMPLES_FILENAME,
                                          hg_model_hub_name=_HG_MODEL_HUB_NAME, train=_TRAIN,
                                          train_bs=_BATCH_SIZE, test_bs=_BATCH_SIZE, epoch=_EPOCHS)


if __name__ == '__main__':
    main()
