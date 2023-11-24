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
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_w_metadata_after_p4.jsonl",
    help='Path to testdata.'
)
parser.add_argument(
    '--predictions_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_w_metadata_after_p4.jsonl",
    help='Path to output file for predictions.'
)
parser.add_argument(
    '--scores_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_w_metadata_after_p4_scores_binary.json",
    help='Path to output file for scores.'
)
args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_PREDICTIONS_OUTPUT_PATH = args.predictions_output_path
_SCORES_OUTPUT_PATH = args.scores_output_path
_CLIENT = openai.OpenAI(
    api_key=args.key,
    timeout=10
)


def main():
    print("Loading test data..")
    # input_data = utils.load_jsonl_file(_TEST_SET_PATH)
    # predictions = prompt_scorer_openai.prompt_openai_model(input_data[:100], _CLIENT, properties.Dataset.AVERITEC)
    # utils.save_jsonl_file(predictions, _PREDICTIONS_OUTPUT_PATH)

    predictions = utils.load_jsonl_file(_PREDICTIONS_OUTPUT_PATH, dataclass=properties.OpenAIResponse)

    scores = prompt_scorer_openai.evaluate_openai_output(predictions, ignore_labels=["not enough evidence",
                                                                                     "conflicting evidence/cherrypicking"])
    with open(_SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics_dict["confusion_metrics"], display_labels=["support", 'refute', "nei"])
    # cm_display.plot()
    # plt.savefig('./results/gpt3.5_confusion_metrics_4th_label_excl.png')


if __name__ == '__main__':
    main()
