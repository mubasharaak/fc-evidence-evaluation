import json
import properties


def load_data(path: str):
    """Loads data from path."""
    with open(path, "r", encoding="utf-8") as file:
        test_dataset = json.load(file)

    return test_dataset


def read_fever_dataset(file_path: str):
    claims = []
    evidences = []
    labels = []
    with open(file_path) as f:
        for line in f:
            line_loaded = json.loads(line)
            for val in list(line_loaded[1].values()):
                claim = val["claim"]
                if val["label"] == 'NOT ENOUGH INFO':
                    continue
                else:
                    label = properties.LABEL_DICT[properties.Label(val["label"].lower())]
                for evidence_tuple in val["evidence"]:
                    if len(evidence_tuple) == 3:
                        evidence = evidence_tuple[2]
                        labels.append(label)
                        claims.append(claim)
                        evidences.append(evidence)
                    else:
                        continue

    return claims, evidences, labels


def read_averitec_dataset(file_path):
    # load file
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    claims = []
    qa_pairs = []
    labels = []
    # iterate
    for entry in dataset:
        if entry["label"] == "Conflicting Evidence/Cherrypicking":
            continue

        claims.append(entry["claim"])
        labels.append(properties.LABEL_DICT[properties.Label(entry["label"].lower())])

        qa_pair = ""
        for qa in entry["questions"]:
            qa_pair += (qa["question"] + " ")
            for a in qa["answers"]:
                qa_pair += (a["answer"] + " ")
                if a["answer_type"] == "Boolean":
                    qa_pair += ("." + a["boolean_explanation"] + ". ")
        qa_pairs.append(qa_pair)

    return claims, qa_pairs, labels
