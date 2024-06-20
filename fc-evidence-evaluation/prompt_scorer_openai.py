import copy
import json
import time

import openai
from sklearn import metrics
from sklearn.metrics import f1_score

import properties
import scorer_utils

_SEED = 10
_MODEL = "gpt-3.5-turbo-1106"
_MAX_TOKENS = 3000
_IGNORE_LABELS_DEFAULT = ["conflicting evidence/cherrypicking"]


def _query_openai(prompt: str, client, keys=None, model: str = None, seed=_SEED, max_tokens=_MAX_TOKENS,
                  response_format="json_object"):
    prompting_model = model if model else _MODEL
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=prompting_model,
        max_tokens=max_tokens,
        response_format={"type": response_format},
        seed=seed,
        temperature=0,
    )


def _get_response_text(response: openai.types.chat.chat_completion.ChatCompletion):
    return response.choices[0].message.content


def _process_output(dataset_sample: properties.AveritecEntry,
                    response: openai.types.chat.chat_completion.ChatCompletion):
    return properties.OpenAIResponse(claim=dataset_sample.claim, evidence=dataset_sample.evidence,
                                     response=_get_response_text(response),
                                     gold=dataset_sample.label.lower(), id=dataset_sample.id)


def _averitec_qa_to_str(evidence: properties.AveritecQA):
    evidence_as_str = (evidence.question + "? ").replace("??", "?")  # sometimes question mark or fullstop missing
    for a in evidence.answers:
        evidence_as_str += (a.answer + ". ").replace("..", ".")
        if a.answer_type.lower() == "boolean":
            evidence_as_str += (a.boolean_explanation + ". ").replace("..", ".")
    return evidence_as_str


def _prepare_prompt(dataset_sample: properties.AveritecEntry, prediction: properties.AveritecEntry,
                    prompt_type: properties.PromptTypes):
    """Formats prompt using dataset sample as input."""
    if type(dataset_sample.evidence) == properties.AveritecQA:
        return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim,
                                                             " ".join([_averitec_qa_to_str(e) for e in
                                                                       dataset_sample.evidence]))
    else:
        if prompt_type in [properties.PromptTypes.ATOMIC_REFERENCE_FACTS,
                           properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL]:
            return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim,
                                                                 dataset_sample.evidence,
                                                                 prediction.evidence)
        else:
            return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim, dataset_sample.evidence)


def _get_system_prediction(sample: properties.AveritecEntry, predictions: list[properties.AveritecEntry]):
    for entry in predictions:
        if entry.claim.lower().strip() == sample.claim.lower().strip():
            return entry
    return None


def calculate_prediction_scores(preds: list[properties.OpenAIResponse], prompt_type: properties.PromptTypes) -> list[
    properties.OpenAIResponse]:
    predictions_w_scores = []
    for pred in preds:
        if prompt_type == properties.PromptTypes.ATOMIC_FACTS:
            predictions_w_scores.append(calculate_atomic_score_openai(pred))
        elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS:
            predictions_w_scores.append(calculate_atomic_score_openai_response(pred))
        elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
            predictions_w_scores.append(calculate_atomic_score_prec_recall_openai_response(pred))

    return predictions_w_scores


def prompt_openai_model(dataset: list[properties.AveritecEntry], predictions: list[properties.AveritecEntry],
                        prompt_type: properties.PromptTypes, client, match_system_preds=True, model: str = None) -> \
list[
    properties.OpenAIResponse]:
    """Prompts OpenAI models."""
    responses = []
    for i, sample in enumerate(dataset):
        print("running sample")
        if match_system_preds:
            # search in predictions for matching prediction
            pred = _get_system_prediction(sample, predictions)
            if not pred:
                responses.append("")
                continue
        else:
            pred = predictions[i]
        prompt = _prepare_prompt(sample, pred, prompt_type)
        while True:
            try:
                responses.append(
                    _process_output(sample, _query_openai(prompt, client, response_format="json_object", model=model)))
                break
            except openai.APITimeoutError as e:
                print(e)
                time.sleep(60)
                pass
    return responses


def calculate_atomic_score_openai_response(response_openai):
    try:
        response_openai_copy = copy.deepcopy(response_openai)
        response = json.loads(response_openai.response)
        response_openai_copy.response = response
        if ("refute" in response and response["refute"] > 0) or (
                "contradict" in response and response["contradict"] > 0):
            # evidence clearly contradicts a sub-fact of the claim
            response_openai_copy.response['score'] = 1
        elif "supports" in response:
            response_openai_copy.response['score'] = response["supports"] / (
                    response["supports"] + response["not enough information"])
        else:
            response_openai_copy.response['score'] = response["support"] / (
                    response["support"] + response["not enough information"])
        return response_openai_copy
    except Exception as e:
        return 0


def calculate_atomic_score_openai(response_openai):
    response_openai_copy = copy.deepcopy(response_openai)
    try:
        response = json.loads(response_openai.response)
        response_openai_copy.response = response
        response_openai_copy.response['score'] = response["support"] / response["facts count"]
    except Exception:
        response_openai_copy.response['score'] = None
    return response_openai_copy


def calculate_atomic_score_prec_recall_openai_response(response_openai):
    response_openai_copy = copy.deepcopy(response_openai)
    try:
        if type(response_openai.response) == str:
            response = json.loads(response_openai.response)
        else:
            response = response_openai.response
        response_openai_copy.response = response
        response_openai_copy.response['precision'] = response["support predicted evidence"] / response[
            "facts count predicted evidence"]
        response_openai_copy.response['recall'] = response["support reference evidence"] / response[
            "facts count reference evidence"]
    except Exception:
        response_openai_copy.response['precision'] = None
        response_openai_copy.response['recall'] = None
    return response_openai_copy


def calculate_atomic_score(response: dict):
    try:
        return response["support"] / response["facts count"]
    except Exception:
        return 0


def calculate_atomic_score_prec_recall(response: dict):
    try:
        prec = response["support predicted evidence"] / response["facts count predicted evidence"]
        recall = response["support reference evidence"] / response["facts count reference evidence"]
        return prec, recall
    except Exception:
        return None, None


def evaluate_openai_output(output_all, prompt_type: properties.PromptTypes, ignore_labels=_IGNORE_LABELS_DEFAULT,
                           is_two_classes=False):
    output = [response for response in output_all if response.gold.lower() not in ignore_labels]
    if prompt_type in [properties.PromptTypes.ATOMIC_FACTS, properties.PromptTypes.ATOMIC_REFERENCE_FACTS]:
        # calculate output score
        scores = [calculate_atomic_score(json.loads(pred.response)) for pred in output]
        return sum(scores) / len(scores)
    elif prompt_type == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
        if type(output[0].response) == str:
            scores = [calculate_atomic_score_prec_recall(json.loads(pred.response)) for pred in output]
        else:
            scores = [calculate_atomic_score_prec_recall(pred.response) for pred in output]
        prec_scores = [x[0] for x in scores if x[0]]
        recall_scores = [x[1] for x in scores if x[1]]
        return sum(prec_scores) / len(prec_scores), sum(recall_scores) / len(recall_scores)

    # map output to labels
    pred_labels = []
    gold_labels = []
    for response in output:
        try:
            pred_label = scorer_utils.map_label(response.response, is_two_classes)
            if response.gold.lower() not in ignore_labels and pred_label in range(3):
                pred_labels.append(pred_label)
                gold_labels.append(properties.LABEL_DICT[properties.Label(response.gold.lower())])
        except Exception as e:
            print("Exception {} for the following claim: {}".format(e, response.claim))

    # calculate metrics (F1 micro/macro) todo replace this by own confusion matrix based on probabilities
    f1_micro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='micro')
    f1_macro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='macro')

    confusion_matrix = metrics.confusion_matrix(gold_labels, pred_labels)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_metrics": confusion_matrix.tolist()}
