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
    '--key',
    default="",
    help='Key for OpenAI API.'
)
parser.add_argument(
    '--test_set_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_dev.json",
    help='Path to testdata.'
)
parser.add_argument(
    '--predictions_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_dev.jsonl",
    help='Path to output file for predictions.'
)
parser.add_argument(
    '--scores_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_dev_scores_binary.json",
    help='Path to output file for scores.'
)
parser.add_argument(
    '--only_evaluate_no_prediction',
    default=False,
    action="store_true",
    help='Given predictions_output_path load predictions for evaluation.'
)
args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_PREDICTIONS_OUTPUT_PATH = args.predictions_output_path
_SCORES_OUTPUT_PATH = args.scores_output_path
_ONLY_EVALUATE = args.only_evaluate_no_prediction
_CLIENT = openai.OpenAI(
    api_key=args.key,
    timeout=10
)


def main():
    if _ONLY_EVALUATE:
        # Given predictions_output_path load predictions for evaluation
        predictions = utils.load_jsonl_file(_PREDICTIONS_OUTPUT_PATH, dataclass=properties.OpenAIResponse)
    else:
        # predict using OpenAI API and store results
        if properties.Dataset.AVERITEC.value not in _TEST_SET_PATH.lower():
            return None
        elif any(entry in _TEST_SET_PATH for entry in properties.AVERITEC_INIT_FILES):
            input_data = utils.load_averitec_base(_TEST_SET_PATH)
        else:
            input_data = utils.load_jsonl_file(_TEST_SET_PATH, properties.AveritecEntry)

        predictions = prompt_scorer_openai.prompt_openai_model(input_data[:1], _CLIENT, properties.Dataset.AVERITEC)
        utils.save_jsonl_file(predictions, _PREDICTIONS_OUTPUT_PATH)

    scores = prompt_scorer_openai.evaluate_openai_output(predictions, ignore_labels=["not enough evidence",
                                                                                     "conflicting evidence/cherrypicking"])
    with open(_SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics_dict["confusion_metrics"], display_labels=["support", 'refute', "nei"])
    # cm_display.plot()
    # plt.savefig('./results/gpt3.5_confusion_metrics_4th_label_excl.png')


if __name__ == '__main__':
    main()
