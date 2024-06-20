import os.path
from typing import Tuple

import openai
import pandas as pd
from scipy import stats

import prompt_scorer_openai
import properties
import utils

_DATA_MAJORITY_VOTING_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_majority.csv"
_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/manual_eval_subset"
_PROMPT_TYPE = properties.PromptTypes("atomic_reference_prec_recall")
_OUTPUT_FILE = "prompt_scorer_{}.jsonl".format(_PROMPT_TYPE.value)
_CORRELATION_OUPUT_FILE = "correlation_coefficients_{}.csv".format(_PROMPT_TYPE.value)
_LOAD_RESULTS = False

_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()
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
                                                   evidence=row['reference evidence'],
                                                   id=row['id']
                                                   ))
        system_pred.append(properties.AveritecEntry(claim=row['claim'],
                                                    label=row['label_majority'],
                                                    evidence=row['predicted evidence'],
                                                    id=row['id']
                                                    ))
    return input_data, system_pred


def _calc_correlation(test_df: pd.DataFrame, results: list[properties.OpenAIResponse],
                      comparison_dim: properties.EvaluationDimensions, score_type=None,
                      ) -> Tuple[int, int]:
    """
    Calculates spearman correlation and pearson correlation
    """
    x = []
    y = []
    for pred in results:
        # get the correct entry from dataframe row
        row = test_df[test_df['id'] == pred.id]
        x.append(row[comparison_dim.value].values[0])

        # get score
        if score_type == properties.ScoreMetrics.PRECISION:
            score = pred.response['precision']
        elif score_type == properties.ScoreMetrics.RECALL:
            score = pred.response['recall']
        else:
            score = pred.response['score']
        y.append(score)

    return stats.spearmanr(x, y).correlation, stats.pearsonr(x, y).statistic


def _run_prompt_scorer(test_df: pd.DataFrame, prompt_type: properties.PromptTypes) -> list[properties.OpenAIResponse]:
    # prepare data
    input_data, system_predictions = _prepare_dataset(test_df)

    # run prompt scorer
    return prompt_scorer_openai.prompt_openai_model(input_data, system_predictions, prompt_type, _CLIENT,
                                                    match_system_preds=False)


def _calc_correlation_append_results(reference: pd.DataFrame, predictions: list[properties.OpenAIResponse],
                                     results_df: pd.DataFrame, score_type: properties.ScoreMetrics,
                                     comparison_dim: properties.EvaluationDimensions):
    spearman_corr, pearson_corr = _calc_correlation(test_df=reference, results=predictions,
                                                    comparison_dim=comparison_dim, score_type=score_type)
    new_row = {
        'scorer': _PROMPT_TYPE.value,
        'metrics': score_type.value,
        'dimension': comparison_dim.value,
        'spearman_corr': spearman_corr,
        'pearson_corr': pearson_corr
    }
    return pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


def _calc_correlation_atomic_reference_based(reference: pd.DataFrame, prediction: list[properties.OpenAIResponse],
                                             results_df: pd.DataFrame):
    # Coverage
    results_df = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                            results_df=results_df,
                                                                            comparison_dim=properties.EvaluationDimensions(
                                                                                "semantic_coverage"))
    # Coherence
    results_df = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                            results_df=results_df,
                                                                            comparison_dim=properties.EvaluationDimensions(
                                                                                "coherence"))
    # Redundancy
    results_df = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                            results_df=results_df,
                                                                            comparison_dim=properties.EvaluationDimensions(
                                                                                "redundancy"))
    # Consistency
    results_df = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                            results_df=results_df,
                                                                            comparison_dim=properties.EvaluationDimensions(
                                                                                "consistency"))
    # Verdict agreement
    results_df = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                            results_df=results_df,
                                                                            comparison_dim=properties.EvaluationDimensions(
                                                                                "verdict_agreement"))
    # NEI disagreement
    return _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference, prediction=prediction,
                                                                      results_df=results_df,
                                                                      comparison_dim=properties.EvaluationDimensions(
                                                                          "nei_disagreement"))


def _calc_correlation_atomic_reference_based_prec_recall_split(reference: pd.DataFrame,
                                                               prediction: list[properties.OpenAIResponse],
                                                               results_df: pd.DataFrame,
                                                               comparison_dim: properties.EvaluationDimensions):
    # precision
    results_df = _calc_correlation_append_results(reference=reference, predictions=prediction,
                                                  results_df=results_df,
                                                  score_type=properties.ScoreMetrics.PRECISION,
                                                  comparison_dim=comparison_dim)

    # recall
    return _calc_correlation_append_results(reference=reference, predictions=prediction,
                                            results_df=results_df,
                                            score_type=properties.ScoreMetrics.RECALL,
                                            comparison_dim=comparison_dim)


def main():
    # load csv file with majority voting
    df = pd.read_csv(_DATA_MAJORITY_VOTING_PATH)

    # load results file and add a new row r = (scorer, spearman, pearson)
    corr_results = pd.read_csv(os.path.join(_OUTPUT_DIR_PATH, _CORRELATION_OUPUT_FILE))

    # run scorer and save results
    if _LOAD_RESULTS:
        model_results_scores = utils.load_jsonl_file(os.path.join(_OUTPUT_DIR_PATH, _OUTPUT_FILE),
                                                     dataclass=properties.OpenAIResponse)
    else:
        model_results = _run_prompt_scorer(df, prompt_type=_PROMPT_TYPE)
        model_results_scores = prompt_scorer_openai.calculate_prediction_scores(model_results, _PROMPT_TYPE)
        utils.save_jsonl_file(model_results_scores, os.path.join(_OUTPUT_DIR_PATH, _OUTPUT_FILE))

    # calculate scores and correlations
    corr_results = _calc_correlation_atomic_reference_based(reference=df, prediction=model_results_scores,
                                                            results_df=corr_results)

    # save results
    corr_results.to_csv(os.path.join(_OUTPUT_DIR_PATH, _CORRELATION_OUPUT_FILE), index=False)


if __name__ == '__main__':
    main()
