from typing import Tuple

import openai
import pandas as pd

import prompt_scorer_openai
import properties

_DATA_MAJORITY_VOTING_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_majority.csv"
_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()
_PROMPT_TYPE = properties.PromptTypes("atomic")
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)


def _prepare_dataset(test_df: pd.DataFrame) -> Tuple[list[properties.AveritecEntry], list[properties.AveritecEntry]]:
    input_data = []
    system_pred = []

    for i, row in test_df.iterrows():
        input_data.append(properties.AveritecEntry(claim=row['claim'],
                                                   label=row['gold label'],
                                                   evidence=row['reference evidence']))
        system_pred.append(properties.AveritecEntry(claim=row['claim'],
                                                    label=row['label_majority'],
                                                    evidence=row['predicted evidence']))
    return input_data, system_pred


def _calc_correlation(test_df: pd.DataFrame, results: list[properties.OpenAIResponse]) -> list[
    properties.AveritecEntry]:
    pass


def _run_prompt_scorer(test_df: pd.DataFrame) -> list[properties.OpenAIResponse]:
    # prepare data
    input_data, system_predictions = _prepare_dataset(test_df)

    # run prompt scorer
    return prompt_scorer_openai.prompt_openai_model(input_data, system_predictions, _PROMPT_TYPE, _CLIENT,
                                                    match_system_preds=False)


def main():
    # load csv file with majority voting
    df = pd.read_csv(_DATA_MAJORITY_VOTING_PATH)

    # run scorer if needed
    model_results = _run_prompt_scorer(df)

    # calculate scores and correlations
    model_results_scores = prompt_scorer_openai.calculate_prediction_scores(model_results, _PROMPT_TYPE)
    corr_results = _calc_correlation(df, model_results_scores)

    # todo save results


if __name__ == '__main__':
    main()
