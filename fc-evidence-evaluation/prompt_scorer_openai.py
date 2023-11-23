import os

import openai
from sklearn import metrics
from sklearn.metrics import f1_score

import properties
import scorer_utils

_DICT_ENTRY = {
    "claim": "",
    "response": "",
    "gold": "",
}
_SEED = 10
_MODEL = "gpt-3.5-turbo-1106"
_MAX_TOKENS = 300
_JSON_PROMPT = "{}. Generate the output in json format with the keys {}."
FEVER_DATASET_PATH = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")


def query_openai(prompt: str, client, keys=None, seed=_SEED, model=_MODEL, max_tokens=_MAX_TOKENS, response_format="json_object"):
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


def _process_output(dataset_sample: dict, response: openai.types.chat.chat_completion.ChatCompletion):
    entry = _DICT_ENTRY.copy()
    entry["claim"] = dataset_sample["claim"]
    entry["response"] = _get_response_text(response)
    entry["gold"] = dataset_sample["label"].lower()
    return entry


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
        entry = _DICT_ENTRY.copy()
        entry["claim"] = claim
        entry["response"] = response
        entry["gold"] = label.lower()
        output_dict.append(entry)
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
        entry = _DICT_ENTRY.copy()
        entry["claim"] = test_expl["claim"]
        entry["response"] = response
        entry["gold"] = test_expl["label"].lower()
        output_dict.append(entry)


def prepare_averitec_prompt(averitec_sample):
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


def prompt_openai_model(dataset: list, client, dataset_name=properties.Dataset):
    """Prompts OpenAI models."""
    responses = []
    for sample in dataset:
        print("running sample")
        # prepare prompt
        if dataset_name == properties.Dataset.AVERITEC:
            prompt = prepare_averitec_prompt(sample)
        elif dataset_name == properties.Dataset.FEVER:
            return responses
        else:
            return responses
        # query OpenAI
        responses.append(_process_output(sample, query_openai(prompt, client, response_format="text")))
        print(responses[-1]["response"])
    return responses


def evaluate_openai_output(output):
    # map output to labels
    pred = []
    gold = []
    for response in output:
        try:
            if response["gold"] != "conflicting evidence/cherrypicking":
                pred.append(scorer_utils.map_label(response["response"]))
                gold.append(properties.LABEL_DICT[properties.Label(response["gold"].lower())])
        except Exception:
            print(f"{scorer_utils.map_label(response['response'])}")
            print(f"{properties.LABEL_DICT[properties.Label(response['gold'].lower())]}")

    # calculate metrics (F1 micro/macro) todo replace this by own confusion matrix based on probabilities
    f1_micro = f1_score(y_true=gold, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=gold, y_pred=pred, average='macro')

    confusion_matrix = metrics.confusion_matrix(gold, pred)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_metrics": confusion_matrix.tolist()}
