import argparse
import json

import openai

import prompt_scorer_openai
import properties
import utils

parser = argparse.ArgumentParser(
    description='Prompt Scorer OpenAI arguments'
)
parser.add_argument(
    '--test_set_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_test.json",
    help='Path to testdata.'
)
parser.add_argument(
    '--predictions_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5_atomic/averitec_test.jsonl",
    help='Path to output file for predictions.'
)
parser.add_argument(
    '--scores_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5_atomic/averitec_test_scores.json",
    help='Path to output file for scores.'
)
parser.add_argument(
    '--only_evaluate_no_prediction',
    default=False,
    action="store_true",
    help='Given predictions_output_path load predictions for evaluation.'
)
parser.add_argument(
    '--prompt_type',
    default="atomic",
    choices=[prompt.value for prompt in properties.PromptTypes],
    type=str.lower
)
args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_PREDICTIONS_OUTPUT_PATH = args.predictions_output_path
_SCORES_OUTPUT_PATH = args.scores_output_path
_ONLY_EVALUATE = args.only_evaluate_no_prediction
_PROMPT_TYPE = properties.PromptTypes(args.prompt_type)

_KEY = open('/Users/user/Desktop/openai_key.txt', 'r').read()
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)


def main():
    if _ONLY_EVALUATE:
        # Given predictions_output_path load predictions for evaluation
        predictions = utils.load_jsonl_file(_PREDICTIONS_OUTPUT_PATH, dataclass=properties.OpenAIResponse)
    else:
        # predict using OpenAI API and store results
        if properties.Dataset.FEVER.value in _TEST_SET_PATH.lower():
            input_data = utils.load_fever(_TEST_SET_PATH)
        elif any(entry in _TEST_SET_PATH.lower() for entry in properties.AVERITEC_INIT_FILES):
            input_data = utils.load_averitec_base(_TEST_SET_PATH)
        else:
            # Averitec with metadata
            input_data = utils.load_jsonl_file(_TEST_SET_PATH, properties.AveritecEntry)
        predictions = prompt_scorer_openai.prompt_openai_model(input_data[:100], _PROMPT_TYPE, _CLIENT)
        utils.save_jsonl_file(predictions, _PREDICTIONS_OUTPUT_PATH)

    # TODO continue by refactoring evaluation, e.g. output of "atomic" prompt
    scores = prompt_scorer_openai.evaluate_openai_output(predictions, _PROMPT_TYPE,
                                                         ignore_labels=["conflicting evidence/cherrypicking",
                                                                        "not enough information", "nei",
                                                                        "not enough info"])
    with open(_SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics_dict["confusion_metrics"], display_labels=["support", 'refute', "nei"])
    # cm_display.plot()
    # plt.savefig('./results/gpt3.5_confusion_metrics_4th_label_excl.png')


if __name__ == '__main__':
    main()
