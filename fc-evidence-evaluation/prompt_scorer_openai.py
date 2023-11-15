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
_MAX_TOKENS = 1000

fever_dataset_path = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")


# if USE_CHATGPT:
#     response = openai.ChatCompletion.create(
#       model="gpt-3.5-turbo",
#       messages=[
#         {"role": "system", "content": init_prompt},
#       ],
#       max_tokens=100
#     )
#     entry = DICT_ENTRY.copy()
#     entry["claim"] = "Pfizer is “managed by Black Rock (sic) finances. Who, by chance, manages the finances of the
#     Open Foundation Company (SOROS FOUNDATION)!"
#     entry["response"] = response
#     print(f"Model output: {entry['response'].choices[0]['message']['content']}")
#
# else:
#     response = openai.Completion.create(
#       model="gpt-3.5-turbo",
#       prompt=init_prompt,
#       temperature=0,
#       max_tokens=100,
#       top_p=1,
#       frequency_penalty=0.0,
#       presence_penalty=0.0,
#       stop=["\n"]
#     )


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


def prompt_openai_model(dataset):
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


def evaluate_openai_output(output):
    # map output to labels
    pred = []
    gold = []
    for response in output:
        try:
            if response["gold"] != "conflicting evidence/cherrypicking":
                pred.append(scorer_utils.map_label(response["response"]["choices"][0]['message']['content']))
                gold.append(properties.LABEL_DICT[properties.Label(response["gold"].lower())])
        except Exception:
            print(f"{scorer_utils.map_label(response['response']['choices'][0]['message']['content'])}")
            print(f"{properties.LABEL_DICT[properties.Label(response['gold'].lower())]}")

    # calculate metrics (F1 micro/macro) todo replace this by own confusion matrix based on probabilities
    f1_micro = f1_score(y_true=gold, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=gold, y_pred=pred, average='macro')

    confusion_matrix = metrics.confusion_matrix(gold, pred)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "confusion_metrics": confusion_matrix}
