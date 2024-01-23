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


def _query_openai(prompt: str, client, keys=None, seed=_SEED, model=_MODEL, max_tokens=_MAX_TOKENS,
                  response_format="json_object"):
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        max_tokens=max_tokens,
        response_format={"type": response_format},
        seed=seed,
    )


def _get_response_text(response: openai.types.chat.chat_completion.ChatCompletion):
    return response.choices[0].message.content


def _process_output(dataset_sample: properties.AveritecEntry,
                    response: openai.types.chat.chat_completion.ChatCompletion):
    return properties.OpenAIResponse(dataset_sample.claim, dataset_sample.evidence, _get_response_text(response),
                                     dataset_sample.label.lower())


def _averitec_qa_to_str(evidence: properties.AveritecQA):
    evidence_as_str = (evidence.question + "? ").replace("??", "?")  # sometimes question mark or fullstop missing
    for a in evidence.answers:
        evidence_as_str += (a.answer + ". ").replace("..", ".")
        if a.answer_type.lower() == "boolean":
            evidence_as_str += (a.boolean_explanation + ". ").replace("..", ".")
    return evidence_as_str


def _prepare_prompt(dataset_sample: properties.AveritecEntry, prompt_type: properties.PromptTypes):
    """Formats prompt using dataset sample as input."""
    if type(dataset_sample.evidence) == properties.AveritecQA:
        return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim,
                                                             " ".join([_averitec_qa_to_str(e) for e in
                                                                       dataset_sample.evidence]))
    else:
        return properties.PROMPT_MAPPING[prompt_type].format(dataset_sample.claim, dataset_sample.evidence)


def prompt_openai_model(dataset: list, prompt_type: properties.PromptTypes, client):
    """Prompts OpenAI models."""
    responses = []
    for sample in dataset:
        print("running sample")
        # try:
        prompt = _prepare_prompt(sample, prompt_type)
        while True:
            try:
                responses.append(_process_output(sample, _query_openai(prompt, client, response_format="json_object")))
                break
            except openai.APITimeoutError as e:
                print(e)
                time.sleep(10)
                pass
        # except Exception as e:
        #     print(e)
        #     continue
    return responses


def calculate_atomic_score(response: dict):
    try:
        if ("refute" in response and response["refute"] > 0) or (
                "contradicts" in response and response["contradicts"] > 0):
            # evidence clearly contradicts a sub-fact of the claim
            return 1
        elif "supports" in response:
            return response["supports"] / (response["supports"] + response["not enough information"])
        else:
            return response["support"] / (response["support"] + response["not enough information"])
    except Exception as e:
        return 0


def evaluate_openai_output(output_all, prompt_type: properties.PromptTypes, ignore_labels=_IGNORE_LABELS_DEFAULT):
    output = [response for response in output_all if response.gold.lower() not in ignore_labels]
    if prompt_type == properties.PromptTypes.ATOMIC_FACTS:
        # calculate output score
        scores = [calculate_atomic_score(json.loads(pred.response)) for pred in output]
        return sum(scores) / len(scores)
    # map output to labels
    pred_labels = []
    gold_labels = []
    for response in output:
        try:
            pred_label = scorer_utils.map_label(response.response)
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
