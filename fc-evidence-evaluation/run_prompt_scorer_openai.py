import argparse
import json
import properties
import openai
import utils

import prompt_scorer_openai

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
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_test.json",
    help='Path to testdata.'
)
parser.add_argument(
    '--predictions_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_test.jsonl",
    help='Path to output file for predictions.'
)
parser.add_argument(
    '--scores_output_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/gpt3.5/averitec_test_scores.json",
    help='Path to output file for scores.'
)
args = parser.parse_args()
_TEST_SET_PATH = args.test_set_path
_PREDICTIONS_OUTPUT_PATH = args.predictions_output_path
_SCORES_OUTPUT_PATH = args.scores_output_path
_CLIENT = openai.OpenAI(
    api_key=args.key,
)


def main():
    print("Loading test data..")
    with open(_TEST_SET_PATH, encoding="utf-8") as file:
        TESTSET = json.load(file)

    predictions = prompt_scorer_openai.prompt_openai_model(TESTSET[:1], _CLIENT, properties.Dataset.AVERITEC)
    utils.save_jsonl_file(predictions, _PREDICTIONS_OUTPUT_PATH)

    scores = prompt_scorer_openai.evaluate_openai_output(predictions)
    with open(_SCORES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=5)

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics_dict["confusion_metrics"], display_labels=["support", 'refute', "nei"])
    # cm_display.plot()
    # plt.savefig('./results/gpt3.5_confusion_metrics_4th_label_excl.png')


if __name__ == '__main__':
    main()
