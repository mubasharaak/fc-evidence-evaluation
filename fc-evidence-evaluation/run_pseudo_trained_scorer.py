import argparse

import evaluate

import nli_scorer
import properties

parser = argparse.ArgumentParser(
    description='NLI Scorer arguments'
)
parser.add_argument(
    '--training_data_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/AveritecBaseline/data/date"
            "-cleaned.train.augmented.json",
    help='Path to training data for fine-tuning NLI scorer'
)
parser.add_argument(
    '--dev_data_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/AveritecBaseline/data/date"
            "-cleaned.dev.augmented.json",
    help='Path to dev data for fine-tuning NLI scorer'
)
parser.add_argument(
    '--test_data_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/AveritecBaseline/data/date"
            "-cleaned.test.augmented.json",
    help='Path to test data for evaluating fine-tuned NLI scorer'
)
parser.add_argument(
    '--output_dir',
    default="./results/nli_scorer",
    help='Output path for NLI scorer evaluation results.'
)
parser.add_argument(
    '--results_filename',
    default="results.json",
    help='Output path for NLI scorer evaluation results.'
)
parser.add_argument(
    '--samples_filename',
    default="prediction_samples.txt",
    help='Output path for NLI scorer evaluation results.'
)
parser.add_argument(
    '--dataset',
    default="averitec",
    choices=list(properties.Dataset),
    help='Dataset that is used for evaluation.'
)
parser.add_argument(
    '--hf_model',
    default="stanleychu2/roberta-fever",
    # hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
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

_BATCH_SIZE = 2
_EPOCHS = 15
_METRIC = evaluate.load("glue", "mrpc")


def main():
    nli_scorer.run_nli_scorer(_HG_MODEL_HUB_NAME, _DATASET, _TRAIN_DATASET_PATH, _DEV_DATASET_PATH,
                              _TEST_DATASET_PATH, _OUTPUT_DIR, _RESULTS_FILENAME, _SAMPLES_FILENAME)


if __name__ == '__main__':
    main()
