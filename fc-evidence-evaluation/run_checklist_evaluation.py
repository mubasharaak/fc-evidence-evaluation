import json
from utils import load_data
from enum import Enum
import checklist_evaluation
import argparse


class TestType(Enum):
    ROBUST_NOISE = "robustness_noise"
    COHERENCE = "coherence"
    COVERAGE = "coverage"
    SYNONYMS = "synonyms"
    RAND_ORDER = "rand_order"
    SUMMARY = "summary"
    CONTRACTION = "contraction"
    NUM2TEXT = "num2text"
    TEXT2NUM = "text2num"
    FLUENCY_TYPOS = "fluency_typos"
    FLUENCY_WORDS_DROP = "fluency_word_drop"
    ENTITY_SWAP = "entity_swap"
    REDUNDANCY_WORDS = "redundancy_words"
    REDUNDANCY_SENT = "redundancy_sent"
    NUMBER_REPLACE = "number_replace"


parser = argparse.ArgumentParser(
    description='Checklist evaluation arguments'
)
parser.add_argument(
    '--input',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/AveritecBaseline/data/date"
            "-cleaned.train.json",
    help='Path input data for checklist test creation.'
)
parser.add_argument(
    '--output',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/AveritecBaseline/data"
            "/robustness_noise_tests.json",
    help='Output path for created checklist tests.'
)
parser.add_argument(
    '--type',
    default="robustness_noise",
    type=TestType,
    choices=list(TestType),
    help='Type of checklist tests to create.'
)

args = parser.parse_args()
_INPUT_PATH = args.input
_OUTPUT_PATH = args.output
_TEST_TYPE = args.type


# TODO add type hint
def create_checklist_tests(input_data):
    """Creates checklist tests."""
    if _TEST_TYPE == TestType.ROBUST_NOISE:
        return checklist_evaluation.robustness_noise_test(input_data)
    elif _TEST_TYPE == TestType.COHERENCE:
        return checklist_evaluation.coherence_test(input_data)
    elif _TEST_TYPE == TestType.COVERAGE:
        return checklist_evaluation.coverage_drop_evidence_part_test(input_data)
    elif _TEST_TYPE == TestType.SYNONYMS:
        return checklist_evaluation.invariance_synonym_test(input_data)
    elif _TEST_TYPE == TestType.RAND_ORDER:
        return checklist_evaluation.informativeness_random_word_order_test(input_data)
    elif _TEST_TYPE == TestType.SUMMARY:
        return checklist_evaluation.create_sum_test(input_data)
    elif _TEST_TYPE == TestType.CONTRACTION:
        return checklist_evaluation.invariance_contraction_test(input_data)
    elif _TEST_TYPE == TestType.NUM2TEXT:
        return checklist_evaluation.invariance_num2text_test(input_data)
    elif _TEST_TYPE == TestType.TEXT2NUM:
        return checklist_evaluation.invariance_text2num_test(input_data)
    elif _TEST_TYPE == TestType.FLUENCY_TYPOS:
        return checklist_evaluation.fluency_typos_test(input_data)
    elif _TEST_TYPE == TestType.FLUENCY_WORDS_DROP:
        return checklist_evaluation.fluency_drop_words_test(input_data)
    elif _TEST_TYPE == TestType.ENTITY_SWAP:
        return checklist_evaluation.informativeness_entity_swap_test(input_data)
    elif _TEST_TYPE == TestType.REDUNDANCY_WORDS:
        return checklist_evaluation.redundancy_duplicate_words_test(input_data)
    elif _TEST_TYPE == TestType.REDUNDANCY_SENT:
        return checklist_evaluation.redundancy_duplicate_sentence_test(input_data)
    elif _TEST_TYPE == TestType.NUMBER_REPLACE:
        return checklist_evaluation.informativeness_number_change_test(input_data)


def main():
    base_data = load_data(_INPUT_PATH)
    test_samples = create_checklist_tests(base_data)

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(test_samples, file, indent=4)


if __name__ == '__main__':
    main()
