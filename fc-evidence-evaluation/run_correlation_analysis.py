import math
import os.path
from typing import Tuple

import openai
import pandas as pd
from scipy import stats

import prompt_scorer_openai
import properties
import utils

# _DATA_MAJORITY_VOTING_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec/averitec_manual_eval_majority.csv"
_DATA_MAJORITY_VOTING_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/averitec_shared_task/AVERITEC_FC_Evidence_Evaluation_Responses_no_gold.xlsx"
_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/averitec_shared_task"
# _PATH_MODEL_RESULTS_SCORES = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/manual_eval_subset/predictions_cot_gpt-4o-2024-05-13_w_scores.jsonl"

_PROMPT_TYPE = properties.PromptTypes("meteor")
# _PROMPTING_MODEL = "gpt-4o-2024-05-13"
_PROMPTING_MODEL = ""
_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()
_LOAD_RESULTS = False

if _PROMPT_TYPE in [properties.PromptTypes.COT, properties.PromptTypes.ATOMIC_FACTS,
                    properties.PromptTypes.ATOMIC_REFERENCE_FACTS,
                    properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL]:
    _OUTPUT_FILE = "predictions_{}_{}.jsonl".format(_PROMPT_TYPE.value, _PROMPTING_MODEL)
    _CORRELATION_OUPUT_FILE = "correlation_{}_{}.csv".format(_PROMPT_TYPE.value, _PROMPTING_MODEL)
else:
    _CORRELATION_OUPUT_FILE = "correlation_{}.csv".format(_PROMPT_TYPE.value)
    if _PROMPT_TYPE in [properties.PromptTypes.PSEUDO_TRAINED, properties.PromptTypes.REF_TRAINED]:
        _OUTPUT_FILE = "predictions_{}.csv".format(_PROMPT_TYPE.value)
    else:
        _OUTPUT_FILE = "predictions_{}.jsonl".format(_PROMPT_TYPE.value)

_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)


def _prepare_dataset(test_df: pd.DataFrame, prev_results: list[properties.OpenAIResponse]) -> Tuple[
    list[properties.AveritecEntry], list[properties.AveritecEntry]]:
    input_data = []
    system_pred = []
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

    # if error in prompting entry can be None
    y = [0 if (entry is None or math.isnan(entry)) else entry for entry in y]
    x = [0 if (entry is None or math.isnan(entry)) else entry for entry in x]
    return stats.spearmanr(x, y).correlation, stats.pearsonr(x, y).statistic


def _run_prompt_scorer(test_df: pd.DataFrame, prompt_type: properties.PromptTypes,
                       prev_results: list[properties.OpenAIResponse], output_path: str = None) -> list[
    properties.OpenAIResponse]:
    # prepare data
    input_data, system_predictions = _prepare_dataset(test_df, prev_results)
    # subset = 10
    # input_data = input_data[:subset]
    # system_predictions = system_predictions[:subset]

    # run prompt scorer
    if prompt_type == properties.PromptTypes.COT:
        return prompt_scorer_openai.prompt_openai_model(input_data, system_predictions, prompt_type, _CLIENT,
                                                        match_system_preds=False, model=_PROMPTING_MODEL,
                                                        responses_output_path=output_path, logprob=True)
    else:
        return prompt_scorer_openai.prompt_openai_model(input_data, system_predictions, prompt_type, _CLIENT,
                                                        match_system_preds=False, model=_PROMPTING_MODEL,
                                                        responses_output_path=output_path, logprob=False)


def _calc_correlation_append_results(reference: pd.DataFrame, predictions: list[properties.OpenAIResponse],
                                     results_df: pd.DataFrame,
                                     comparison_dim: properties.EvaluationDimensions,
                                     score_type: properties.ScoreMetrics = None, ):
    spearman_corr, pearson_corr = _calc_correlation(test_df=reference, results=predictions,
                                                    comparison_dim=comparison_dim, score_type=score_type)
    new_row = {
        'scorer': _PROMPT_TYPE.value,
        'metrics': score_type.value if score_type else "",
        'dimension': comparison_dim.value,
        'spearman_corr': spearman_corr,
        'pearson_corr': pearson_corr
    }
    return pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


def _calc_correlation_atomic_reference_based(reference: pd.DataFrame, prediction: list[properties.OpenAIResponse],
                                             results_df: pd.DataFrame, prompt_type: properties.PromptTypes,
                                             evaluation_dimensions: list[properties.EvaluationDimensions]):
    results_df_copy = results_df.copy(deep=True)
    for evaluation_dimension in evaluation_dimensions:
        results_df_copy = _calc_correlation_atomic_reference_based_prec_recall_split(reference=reference,
                                                                                     prediction=prediction,
                                                                                     results_df=results_df_copy,
                                                                                     comparison_dim=evaluation_dimension,
                                                                                     prompt_type=prompt_type)
    return results_df_copy


def _calc_correlation_atomic_reference_based_prec_recall_split(reference: pd.DataFrame,
                                                               prediction: list[properties.OpenAIResponse],
                                                               results_df: pd.DataFrame,
                                                               comparison_dim: properties.EvaluationDimensions,
                                                               prompt_type: properties.PromptTypes):
    if prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
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
    else:
        return _calc_correlation_append_results(reference=reference, predictions=prediction,
                                                results_df=results_df,
                                                comparison_dim=comparison_dim,
                                                score_type=None)


def _calculate_scores_baseline_metrics(df: pd.DataFrame, prompt_type: properties.PromptTypes,
                                       results_df: pd.DataFrame = None) -> list[
    properties.OpenAIResponse]:
    results = []
    for i, row in df.iterrows():
        if prompt_type == properties.PromptTypes.METEOR:
            score = utils.calc_meteor(reference=row['reference evidence'], candidate=row['predicted evidence'])
        elif prompt_type == properties.PromptTypes.HMETEOR:
            score = utils.calc_hungarian_meteor(candidate=row['predicted evidence'],
                                                reference=row['reference evidence'])
        elif prompt_type == properties.PromptTypes.BLEU:
            score = utils.calc_bleu(candidate=row['predicted evidence'], reference=row['reference evidence'])
        elif prompt_type == properties.PromptTypes.PSEUDO_TRAINED:
            score = results_df.iloc[i]['score']
        elif prompt_type == properties.PromptTypes.REF_TRAINED:
            score = results_df.iloc[i]['prediction']
        elif prompt_type == properties.PromptTypes.ROUGE:
            score = utils.calc_rouge(candidate=row['predicted evidence'], reference=row['reference evidence'])
        else:
            raise Exception("prompt_type equal {} not found in enum properties.PromptTypes.".format(prompt_type))
        results.append(properties.OpenAIResponse(claim=row['claim'], evidence=row['predicted evidence'],
                                                 response={'reference_evidence': row['reference evidence'],
                                                           'score': score},
                                                 gold=row['gold label'], id=row['id']))

    return results


def main():
    scorer_output_path = os.path.join(_OUTPUT_DIR_PATH, _OUTPUT_FILE)
    # load csv file with majority voting
    if _DATA_MAJORITY_VOTING_PATH.endswith(".csv"):
        df = pd.read_csv(_DATA_MAJORITY_VOTING_PATH)
    elif _DATA_MAJORITY_VOTING_PATH.endswith(".xlsx"):
        df = pd.read_excel(_DATA_MAJORITY_VOTING_PATH, header=0)
    else:
        raise ValueError("Filepath of variable '_DATA_MAJORITY_VOTING_PATH' must end with '.csv' or '.xlsx'.")
    # load results file and add a new row r = (scorer, spearman, pearson)
    corr_results = pd.read_csv(os.path.join(_OUTPUT_DIR_PATH, _CORRELATION_OUPUT_FILE))

    # run scorer and save results
    if _PROMPT_TYPE in [properties.PromptTypes.PSEUDO_TRAINED, properties.PromptTypes.REF_TRAINED]:
        model_results_scores = _calculate_scores_baseline_metrics(df, _PROMPT_TYPE, pd.read_csv(scorer_output_path))
    elif _PROMPT_TYPE in [properties.PromptTypes.METEOR, properties.PromptTypes.ROUGE, properties.PromptTypes.BLEU,
                          properties.PromptTypes.HMETEOR]:
        model_results_scores = _calculate_scores_baseline_metrics(df, _PROMPT_TYPE)
        utils.save_jsonl_file(model_results_scores, scorer_output_path)
    else:
        if _LOAD_RESULTS:
            model_results_scores = utils.load_jsonl_file(scorer_output_path, dataclass=properties.OpenAIResponse)
        else:
            # if prev model_results_scores exist, load
            if os.path.exists(scorer_output_path):
                model_results_scores = utils.load_jsonl_file(scorer_output_path,
                                                             dataclass=properties.OpenAIResponse)
            else:
                model_results_scores = []
            model_results = _run_prompt_scorer(df, prompt_type=_PROMPT_TYPE, prev_results=model_results_scores,
                                               output_path=scorer_output_path)
            utils.save_jsonl_file(model_results, scorer_output_path)

            model_results_scores.extend(
                prompt_scorer_openai.calculate_prediction_scores(df, model_results, _PROMPT_TYPE))
            utils.save_jsonl_file(model_results_scores, scorer_output_path)

    # calculate scores and correlations
    corr_results = _calc_correlation_atomic_reference_based(reference=df, prediction=model_results_scores,
                                                            results_df=corr_results, prompt_type=_PROMPT_TYPE,
                                                            evaluation_dimensions=[
                                                                properties.EvaluationDimensions("semantic_coverage"),
                                                                properties.EvaluationDimensions("coherence"),
                                                                properties.EvaluationDimensions("redundancy"),
                                                                properties.EvaluationDimensions("consistency"),
                                                                properties.EvaluationDimensions("relevance"),
                                                                properties.EvaluationDimensions("verdict_agreement"),
                                                            ])

    # save results
    corr_results.to_csv(os.path.join(_OUTPUT_DIR_PATH, _CORRELATION_OUPUT_FILE), index=False)


if __name__ == '__main__':
    main()
