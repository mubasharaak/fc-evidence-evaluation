import json
import properties
import dacite
import sqlite3
import unicodedata


def load_json_file(path: str):
    """Loads data from path."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


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


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def save_jsonl_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(to_dict(entry), f)
            f.write("\n")


def load_jsonl_file(file_path, dataclass=None):
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for entry in f.readlines():
            if dataclass:
                content.append(dacite.from_dict(data_class=dataclass, data=json.loads(entry)))
            else:
                content.append(json.loads(entry))
    return content


def map_averitec_to_dataclass_format(averitec: dict):
    """Formats Averitec dataset files to match fields specified in properties.AveritecEntry."""
    return dacite.from_dict(data_class=properties.AveritecEntry,
                            data={"claim": averitec["claim"], "label": averitec["label"],
                                  "justification": averitec["justification"],
                                  "evidence": averitec["questions"]})


def load_averitec_base(path: str) -> list[properties.AveritecEntry]:
    """Loads and formats Averitec dataset files (train, test, or dev)."""
    return [map_averitec_to_dataclass_format(entry) for entry in load_json_file(path)]


def map_fever_to_dataclass_format(fever_entry: list):
    """Formats Averitec dataset files to match fields specified in properties.AveritecEntry."""
    mapped_entries = []
    for fever_subentry in iter(fever_entry[1].values()):
        evidence = ". ".join(e[2].strip() for e in fever_subentry["evidence"] if len(e)>2).replace("..", ".")
        mapped_entries.append(dacite.from_dict(data_class=properties.AveritecEntry,
                                               data={"claim": fever_subentry["claim"], "label": fever_subentry["label"],
                                                     "justification": "",
                                                     "evidence": evidence}))
    return mapped_entries


def load_fever(path: str) -> list[properties.AveritecEntry]:
    """Loads and formats Fever files."""
    fever_entries = []
    for entry in load_jsonl_file(path):
        fever_entries.extend(map_fever_to_dataclass_format(entry))
    return fever_entries


def _load_hover_evidence(evidences: list, wiki_db):
    evidence_text = ""
    for e in evidences:
        # e of format: [
        #         "Life Goes On (Fergie song)",
        #         2
        #       ]
        doc = wiki_db.execute("SELECT * FROM documents WHERE id=(?)", (unicodedata.normalize('NFD', e[0]),)).fetchall()[0]
        # retrieve relevant sentence as evidence
        evidence_text += doc.split()[e[1]]
    return evidence_text


def _map_hover_to_dataclass_format(hover_entry: dict, wiki_db) -> properties.AveritecEntry:
    """Formats Hover dataset files to match fields specified in properties.AveritecEntry."""
    return dacite.from_dict(data_class=properties.AveritecEntry,
                            data={"claim": hover_entry["claim"], "label": hover_entry["label"],
                                  "evidence": _load_hover_evidence(hover_entry["supporting_facts"], wiki_db)})


def load_hover(path: str, wiki_db) -> list[properties.AveritecEntry]:
    """Loads and formats Hover dataset."""
    entries = []
    for entry in load_json_file(path):
        entries.append(_map_hover_to_dataclass_format(entry, wiki_db))
    return entries


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return c
