import os
import time

import openai
from sklearn import metrics
from sklearn.metrics import f1_score

import properties
import scorer_utils

_SEED = 10
_MODEL = "gpt-3.5-turbo-1106"
_MAX_TOKENS = 150
_JSON_PROMPT = "{}. Generate the output in json format with the keys {}."
FEVER_DATASET_PATH = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")
_IGNORE_LABELS_DEFAULT = ["conflicting evidence/cherrypicking"]


def query_openai(prompt: str, client, keys=None, seed=_SEED, model=_MODEL, max_tokens=_MAX_TOKENS,
                 response_format="json_object"):
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": _JSON_PROMPT.format(prompt, ", ".join(keys)) if response_format == "json_object" else prompt,
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
    return properties.OpenAIResponse(dataset_sample.claim, _get_response_text(response),
                                     dataset_sample.label.lower())


def prompt_openai_model_fever_submissions(claims, evidences, labels):
    # iterate over test examples (first 10 for testing purposes)
    output_dict = []
    for claim, evidence, label in zip(claims, evidences, labels):
        prompt = properties.BASE_PROMPT
        prompt += "Claim: " + claim + "\n"
        prompt += "Evidence: " + evidence + "\n"
        prompt += "Answer: "

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=_MAX_TOKENS
        )
        output_dict.append(properties.OpenAIResponse(claim, response, label.lower()))
    return output_dict


def prompt_openai_model_deprecated(dataset):
    # iterate over test examples (first 10 for testing purposes)
    output_dict = []
    for test_expl in dataset:
        prompt = properties.BASE_PROMPT
        prompt += "Claim: " + test_expl["claim"] + "\n"
        qa_pair = ""
        for qa in test_expl["questions"]:
            qa_pair += (qa["question"] + " ")
            for a in qa["answers"]:
                qa_pair += (a["answer"] + " ")
                if a["answer_type"] == "Boolean":
                    qa_pair += (a["boolean_explanation"] + ". ")
        prompt += "Evidence: " + qa_pair + "\n"
        prompt += "Answer: "

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=_MAX_TOKENS
        )
        output_dict.append(properties.OpenAIResponse(test_expl["claim"], response, test_expl["label"].lower()))


def prepare_averitec_prompt_deprecated(averitec_sample):
    """Formats prompt using Averitec sample as input."""
    prompt = properties.BASE_PROMPT
    prompt += "Claim: " + averitec_sample["claim"] + "\n"
    qa_pair = ""
    for qa in averitec_sample["questions"]:
        qa_pair += (qa["question"] + " ")
        for a in qa["answers"]:
            qa_pair += (a["answer"] + " ")
            if a["answer_type"] == "Boolean":
                qa_pair += (a["boolean_explanation"] + ". ")
    prompt += "Evidence: " + qa_pair + "\n"
    prompt += "Answer: "
    return prompt


def averitec_qa_to_str(evidence: properties.AveritecQA):
    evidence_as_str = (evidence.question + "? ").replace("??", "?")  # sometimes question mark or fullstop missing
    for a in evidence.answers:
        evidence_as_str += (a.answer + ". ").replace("..", ".")
        if a.answer_type.lower() == "boolean":
            evidence_as_str += (a.boolean_explanation + ". ").replace("..", ".")
    return evidence_as_str


def prepare_prompt(dataset_sample: properties.AveritecEntry):
    """Formats prompt using Averitec sample as input."""
    if type(dataset_sample.evidence) == properties.AveritecQA:
        return properties.BASE_PROMPT.format(dataset_sample.claim,
                                             " ".join([averitec_qa_to_str(e) for e in dataset_sample.evidence]))
    else:
        return properties.BASE_PROMPT.format(dataset_sample.claim, dataset_sample.evidence)


def prompt_openai_model(dataset: list, client):
    """Prompts OpenAI models."""
    responses = []
    for sample in dataset:
        print("running sample")
        # try:
        prompt = prepare_prompt(sample)
        while True:
            try:
                responses.append(_process_output(sample, query_openai(prompt, client, response_format="text")))
                break
            except openai.APITimeoutError as e:
                print(e)
                time.sleep(10)
                pass
        # except Exception as e:
        #     print(e)
        #     continue
    return responses


def evaluate_openai_output(output, ignore_labels=_IGNORE_LABELS_DEFAULT):
    # map output to labels
    pred_labels = []
    gold_labels = []
    for response in output:
        try:
            pred_label = scorer_utils.map_label(response.response)
            if response.gold.lower() not in ignore_labels and pred_label in range(2):
                pred_labels.append(pred_label)
                gold_labels.append(properties.LABEL_DICT[properties.Label(response.gold.lower())])
        except Exception:
            print(f"{scorer_utils.map_label(response['response'])}")
            print(f"{properties.LABEL_DICT[properties.Label(response['gold'].lower())]}")

    # calculate metrics (F1 micro/macro) todo replace this by own confusion matrix based on probabilities
    f1_micro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='micro')
    f1_macro = f1_score(y_true=gold_labels, y_pred=pred_labels, average='macro')

    confusion_matrix = metrics.confusion_matrix(gold_labels, pred_labels)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_metrics": confusion_matrix.tolist()}
