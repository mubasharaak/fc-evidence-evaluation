"""
Script to run previously created checklist tests (create_checklist_evaluation_tests.py) against scorers.
"""
import os.path
import statistics

import openai
import pandas as pd

import prompt_scorer_openai
import properties
import utils

_TESTS_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/checklist_tests"
_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/checklist"
_RESULTS_OUTPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results.csv")
_INIT_DATA_PATH = os.path.join(_TESTS_DIR_PATH, "base_data.json")

_TEST_TYPE = properties.TestType("coherence")
_TEST_FILE_PATH = os.path.join(_TESTS_DIR_PATH, "{}.json".format(_TEST_TYPE.value))

_PROMPT_TYPE = properties.PromptTypes("atomic_reference_prec_recall")
_PROMPTING_MODEL = "gpt-4o-2024-05-13"
_OUTPUT_FILE = "predictions_{}_{}_{}.jsonl".format(_PROMPT_TYPE.value, _PROMPTING_MODEL, _TEST_TYPE.value)

_LOAD_RESULTS = False

_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)


def _prepare_dataset(test_df: pd.DataFrame, prev_results: list[properties.OpenAIResponse]) -> list[
    properties.AveritecEntry]:
    """
    Based on averitec input data, prepares a list of properties.AveritecEntry
    :param test_df:
    :param prev_results:
    :return:
    """
    input_data = []
    prev_claims = [x.claim for x in prev_results]

    for i, row in test_df.iterrows():
        # if model has been previously prompted for this entry -> skip
        if row['claim'] in prev_claims:
            continue
        input_data.append(properties.AveritecEntry(claim=row['claim'],
                                                   label=row['gold label'],
                                                   evidence=row['reference evidence'],
                                                   id=row['id']
                                                   ))
    return input_data


def _run_prompt_scorer(init_data_samples: list[properties.AveritecEntry], test_samples: list[properties.AveritecEntry],
                       prompt_type: properties.PromptTypes,
                       output_path: str) -> list[properties.OpenAIResponse]:
    # run prompt scorer
    return prompt_scorer_openai.prompt_openai_model(init_data_samples, test_samples, prompt_type,
                                                    _CLIENT, match_system_preds=False, model=_PROMPTING_MODEL,
                                                    responses_output_path=output_path)


def _calc_test_results(scorer_results, results_df) -> pd.DataFrame:
    """
    Returns average score across all scored samples. Then compares the obtained score to scores for init_data
    :param scorer_results:
    :param results_df:
    :return:
    """
    if _PROMPT_TYPE == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
        new_rows = [
            {
                'test': _TEST_TYPE.value,
                'scorer': _PROMPT_TYPE.value.replace("_recall", ""),
                'score': statistics.mean([x.response['precision'] for x in scorer_results])
            },
            {
                'test': _TEST_TYPE.value,
                'scorer': _PROMPT_TYPE.value.replace("_prec", ""),
                'score': statistics.mean([x.response['recall'] for x in scorer_results]),
            }
        ]
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        score = 0

    return results_df


def main():
    scorer_output_path = os.path.join(_OUTPUT_DIR_PATH, _OUTPUT_FILE)

    # load initial data and test file
    init_data = utils.load_averitec_base(_INIT_DATA_PATH, extract_evidence=True)
    tests = utils.load_averitec_base(_TEST_FILE_PATH, extract_evidence=True)

    # load results file and add a new row r = (scorer, spearman, pearson)
    results = pd.read_csv(os.path.join(_OUTPUT_DIR_PATH, _RESULTS_OUTPUT_FILE))

    # run scorer and save results
    if _LOAD_RESULTS:
        model_results_scores = utils.load_jsonl_file(scorer_output_path,
                                                     dataclass=properties.OpenAIResponse)
    else:
        if _PROMPT_TYPE in [properties.PromptTypes.METEOR, properties.PromptTypes.ROUGE]:
            # model_results_scores = _calculate_scores_baseline_metrics(tests, _PROMPT_TYPE)
            model_results_scores = None
        else:
            model_results = _run_prompt_scorer(init_data, tests, prompt_type=_PROMPT_TYPE,
                                               output_path=scorer_output_path)
            model_results_scores = prompt_scorer_openai.calculate_prediction_scores(input_data=None,
                                                                                    preds=model_results,
                                                                                    prompt_type=_PROMPT_TYPE)
            utils.save_jsonl_file(model_results_scores, scorer_output_path)

    # calculate changes in scores by comparing to base dataset
    results = _calc_test_results(model_results_scores, results)

    # save results
    results.to_csv(os.path.join(_OUTPUT_DIR_PATH, _RESULTS_OUTPUT_FILE), index=False)


if __name__ == '__main__':
    main()
