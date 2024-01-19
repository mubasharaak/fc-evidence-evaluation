import argparse
import os
import evaluate

import properties
import pseudo_trained_scorer

parser = argparse.ArgumentParser(
    description='NLI Scorer arguments'
)
parser.add_argument(
    '--data_dir',
    default="/scratch/users/k20116188/fc_evidence_evaluation/pseudo_trained_scorer_training_data",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--training_data_file',
    default="train.jsonl",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--dev_data_file',
    default="dev.jsonl",
    help='Path to dev data for reference scorer'
)
parser.add_argument(
    '--test_data_path',
    default="/scratch/users/k20116188/fc_evidence_evaluation/datasets/averitec/averitec_w_metadata_before_p4.jsonl",
    help='Path to test data for evaluating fine-tuned reference scorer'
)
parser.add_argument(
    '--output_dir',
    default="/scratch/users/k20116188/fc_evidence_evaluation/results/pseudo_trained_scorer",
    help='Output path for NLI scorer evaluation results.'
)
parser.add_argument(
    '--results_filename',
    default="results_{}.json",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--samples_filename',
    default="prediction_{}.txt",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--dataset',
    default="vitaminc",     # set to vitaminc if jsonl file with claim, evidence, label entries in dicts.
    choices=list(properties.Dataset),
    help='Dataset that is used for evaluation.'
)
parser.add_argument(
    '--hf_model',
    default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    # hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "stanleychu2/roberta-fever"
    help='Dataset that is used for evaluation.'
)
parser.add_argument(
    '--finetuned_model',
    default="/users/k20116188/projects/averitec/results/deberta_numerical_augmentation",
    help='Path to fine-tuned model.'
)
parser.add_argument(
    '--train',
    default=False,
    action="store_true",
    help='If set, fine-tunes scorer with data specified through --training_data_path'
)

args = parser.parse_args()
_DATA_DIR = args.data_dir
_TRAIN_DATASET_PATH = os.path.join(_DATA_DIR, args.training_data_file)
_DEV_DATASET_PATH = os.path.join(_DATA_DIR, args.dev_data_file)
_TEST_DATASET_PATH = args.test_data_path

_OUTPUT_DIR = args.output_dir
test_file_name = args.test_data_path.split("/")[-1].split(".")[0]
_RESULTS_FILENAME = args.results_filename.format(test_file_name)
_SAMPLES_FILENAME = args.samples_filename.format(test_file_name)

print("Results saved in: {}".format(_RESULTS_FILENAME))

_DATASET = properties.Dataset(args.dataset)

_TRAIN = args.train
if _TRAIN:
    _MODEL_PATH = args.hf_model
else:
    _MODEL_PATH = args.finetuned_model

_BATCH_SIZE = 4
_EPOCHS = 3
_METRIC = evaluate.load("glue", "mrpc")


def main():
    pseudo_trained_scorer.run_nli_scorer(model_path=_MODEL_PATH, dataset=_DATASET,
                                         train_dataset_path=_TRAIN_DATASET_PATH, dev_dataset_path=_DEV_DATASET_PATH,
                                         test_dataset_path=_TEST_DATASET_PATH, output_path=_OUTPUT_DIR,
                                         results_filename=_RESULTS_FILENAME, samples_filenames=_SAMPLES_FILENAME,
                                         train_model=_TRAIN, train_bs=_BATCH_SIZE, test_bs=_BATCH_SIZE, epoch=_EPOCHS)


if __name__ == '__main__':
    main()
