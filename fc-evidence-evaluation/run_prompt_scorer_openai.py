import argparse
import json
import os
import random

import openai
import pymysql

import prompt_scorer_openai
import properties
import utils

parser = argparse.ArgumentParser(
    description='Prompt Scorer OpenAI arguments'
)
parser.add_argument(
    '--test_set_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/pseudo_scorer_training_data/vitaminc_test.jsonl",
    help='Path to testdata.'
)
parser.add_argument(
    '--predictions_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5_pseudo/prediction_vitaminc_test.jsonl",
    help='Path to output file for predictions.'
)
parser.add_argument(
    '--scores_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5_pseudo/results_vitaminc_test.json",
    help='Path to output file for scores.'
)
parser.add_argument(
    '--only_evaluate_no_prediction',
    default=False,
    action="store_true",
    help='Given predictions_output_path load predictions for evaluation.'
)
parser.add_argument(
    '--dataset',
    default="vitaminc",  # set to vitaminc if jsonl file with claim, evidence, label entries in dicts.
    choices=list(properties.Dataset),
    help='Dataset that is used for evaluation.'
)
parser.add_argument(
    '--prompt_type',
    default="cot",
    choices=[prompt.value for prompt in properties.PromptTypes],
    type=str.lower
)
args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_PREDICTIONS_OUTPUT_PATH = args.predictions_output_path
_SCORES_OUTPUT_PATH = args.scores_output_path
_ONLY_EVALUATE = args.only_evaluate_no_prediction
_PROMPT_TYPE = properties.PromptTypes(args.prompt_type)
_DATASET = properties.Dataset(args.dataset)
_IS_HOVER_DATASET = True if "hover" in _TEST_SET_PATH else False

_KEY = open('/Users/user/Desktop/openai_key.txt', 'r').read()
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)
_SEED = 10
_RANDOM_SUBSET = 100
random.seed(_SEED)
_WIKI_DB_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data"
_FEVER_DB_PW = open('/Users/user/Desktop/fever_db_pw.txt', 'r').read()


def _prepare_dataset_samples(claims: list, evidences: list, labels: list):
    samples = []
    for claim, evid, lable in zip(claims, evidences, labels):
        samples.append(properties.AveritecEntry(claim=claim, evidence=evid, label=lable, justification=""))
    return samples


def _load_dataset(dataset, test_dataset_path):
    if dataset == properties.Dataset.FEVER:
        wiki_db = pymysql.connect(host="localhost", port=3306, user="root", password=_FEVER_DB_PW, db="fever").cursor()
        claims, evidences, labels = utils.read_fever_dataset(test_dataset_path, wiki_db)
    elif dataset == properties.Dataset.FEVER_REANNOTATION:
        claims, evidences, labels = utils.read_fever_dataset_reannotation(test_dataset_path)
    elif dataset == properties.Dataset.AVERITEC:
        claims, evidences, labels = utils.read_averitec_dataset(test_dataset_path)
    elif dataset == properties.Dataset.AVERITEC_AFTER_P4:
        claims, evidences, labels = utils.read_averitec_before_after_p4(test_dataset_path)
    elif dataset == properties.Dataset.HOVER:
        wiki_db = utils.connect_to_db(os.path.join(_WIKI_DB_PATH, "hover", 'wiki_wo_links.db'))
        claims, evidences, labels = utils.read_hover_dataset(test_dataset_path, wiki_db)
    elif dataset == properties.Dataset.VITAMINC:
        # also used for train.jsonl and dev.jsonl => all
        claims, evidences, labels = utils.read_vitaminc_dataset(test_dataset_path)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))
    labels = [properties.LABEL_DICT_REVERSE[l] for l in labels]
    return _prepare_dataset_samples(claims, evidences, labels)


def main():
    if _ONLY_EVALUATE:
        # Given predictions_output_path load predictions for evaluation
        predictions = utils.load_jsonl_file(_PREDICTIONS_OUTPUT_PATH, dataclass=properties.OpenAIResponse)
    else:
        input_data = random.sample(_load_dataset(_DATASET, _TEST_SET_PATH), _RANDOM_SUBSET)
        # predict using OpenAI API and store results
        predictions = prompt_scorer_openai.prompt_openai_model(input_data, _PROMPT_TYPE, _CLIENT)
        utils.save_jsonl_file(predictions, _PREDICTIONS_OUTPUT_PATH)

    # TODO continue by refactoring evaluation, e.g. output of "atomic" prompt
    scores = prompt_scorer_openai.evaluate_openai_output(predictions, _PROMPT_TYPE,
                                                         ignore_labels=["conflicting evidence/cherrypicking"],
                                                         is_two_classes=_IS_HOVER_DATASET)
    with open(_SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics_dict["confusion_metrics"], display_labels=["support", 'refute', "nei"])
    # cm_display.plot()
    # plt.savefig('./results/gpt3.5_confusion_metrics_4th_label_excl.png')


if __name__ == '__main__':
    main()
