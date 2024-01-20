import json
import os
import re
import pymysql

import evaluate
import scorer_utils
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import TrainingArguments

import properties
import utils

_MAX_LENGTH = 1024
# fever_dataset_path = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")
_TBL_DELIM = " ; "
_ENTITY_LINKING_PATTERN = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')
_WIKI_DB_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data"

_METRIC = evaluate.load("glue", "mrpc")
_PATH_TOKENIZER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_FEVER_DB_PW = open('/scratch/users/k20116188/fc_evidence_evaluation/credentials/fever_db_pw.txt', 'r').read()


class PseudoTrainedScorerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


def _compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # print("labels: {}".format(labels))
    # print("logits: {}".format(logits))
    # print("predictions: {}".format(predictions))

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions, average='weighted')
    recall = recall_score(y_true=labels, y_pred=predictions, average='weighted')
    f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }


def train(model, training_args, train_dataset, dev_dataset, test_dataset, output_path, do_training=False):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=_compute_metrics,
    )

    if do_training:
        trainer.train()
        trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print("Result_dict: {}".format(result_dict.metrics))
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset, output_path):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=scorer_utils.compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    print(result_dict.metrics)
    return result_dict


def prepare_dataset(claims, evidence, labels, tokenizer):
    data_tokenized = tokenizer(evidence, claims,
                               max_length=_MAX_LENGTH,
                               truncation=True,
                               padding=True, return_tensors="pt")

    # print("data_tokenized: {}".format(data_tokenized))
    return PseudoTrainedScorerDataset(data_tokenized, labels)


def check_dataset(dataset):
    print("Check for NaNs..")
    for i, item in enumerate(dataset):
        print("process ", i)
        for k in item:
            if torch.isnan(item[k]).any():
                print("NaN in item ", i, " ", k)
    print("Check completed")


def run_nli_scorer(model_path: str, dataset: properties.Dataset, train_dataset_path: str, dev_dataset_path: str,
                   test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
                   train_model: bool, train_bs: int, test_bs: int, epoch: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if train_model:
        print("Model will be trained!")
        model.train()

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epoch,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=test_bs,
        warmup_steps=50,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=2500,
        save_steps=2500,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-06,
        fp16=False,  # mixed precision training
        # debug="underflow_overflow",
    )
    # training_args = TrainingArguments(
    #     output_dir=output_path,  # output directory
    #     num_train_epochs=4,  # total number of training epochs
    #     learning_rate=4e-04,
    #     per_device_train_batch_size=4,  # batch size per device during training
    #     gradient_accumulation_steps=2,  # doubles the effective batch_size to 32, while decreasing memory requirements
    #     per_device_eval_batch_size=64,  # batch size for evaluation
    #     warmup_ratio=0.06,  # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     # fp16=True,  # mixed precision training
    #     evaluation_strategy="steps",
    #     eval_steps = 25,
    #     save_steps = 25,
    #     metric_for_best_model = "eval_f1_micro",
    #     save_total_limit = 1,
    #     load_best_model_at_end = True,
    # )

    if dataset == properties.Dataset.FEVER:
        wiki_db = pymysql.connect(host="localhost", port=3306, user="root", password=_FEVER_DB_PW, db="fever").cursor()
        train_claims, train_evidences, train_labels = utils.read_fever_dataset(train_dataset_path, wiki_db)
        test_claims, test_evidences, test_labels = utils.read_fever_dataset(test_dataset_path, wiki_db)
        eval_claims, dev_evidences, eval_labels = utils.read_fever_dataset(dev_dataset_path, wiki_db)
    elif dataset == properties.Dataset.FEVER_REANNOTATION:
        train_claims, train_evidences, train_labels = utils.read_fever_dataset_reannotation(train_dataset_path)
        test_claims, test_evidences, test_labels = utils.read_fever_dataset_reannotation(test_dataset_path)
        eval_claims, dev_evidences, eval_labels = utils.read_fever_dataset_reannotation(dev_dataset_path)
    elif dataset in [properties.Dataset.AVERITEC, properties.Dataset.AVERITEC_AFTER_P4]:
        train_claims, train_evidences, train_labels = utils.read_averitec_dataset(train_dataset_path)
        test_claims, test_evidences, test_labels = utils.read_averitec_dataset(test_dataset_path)
        eval_claims, dev_evidences, eval_labels = utils.read_averitec_dataset(dev_dataset_path)
    elif dataset == properties.Dataset.HOVER:
        wiki_db = utils.connect_to_db(os.path.join(_WIKI_DB_PATH, "hover", 'wiki_wo_links.db'))
        train_claims, train_evidences, train_labels = utils.read_hover_dataset(train_dataset_path, wiki_db)
        test_claims, test_evidences, test_labels = utils.read_hover_dataset(test_dataset_path, wiki_db)
        eval_claims, dev_evidences, eval_labels = utils.read_hover_dataset(dev_dataset_path, wiki_db)
    elif dataset == properties.Dataset.VITAMINC:
        # also used for train.jsonl and dev.jsonl => all
        train_claims, train_evidences, train_labels = utils.read_vitaminc_dataset(train_dataset_path)
        test_claims, test_evidences, test_labels = utils.read_vitaminc_dataset(test_dataset_path)
        eval_claims, dev_evidences, eval_labels = utils.read_vitaminc_dataset(dev_dataset_path)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))

    train_dataset = prepare_dataset(train_claims, train_evidences, train_labels, tokenizer)
    dev_dataset = prepare_dataset(eval_claims, dev_evidences, eval_labels, tokenizer)
    test_dataset = prepare_dataset(test_claims, test_evidences, test_labels, tokenizer)

    check_dataset(train_dataset)
    check_dataset(test_dataset)
    check_dataset(dev_dataset)

    results = train(model, training_args, train_dataset=train_dataset,
                    dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path,
                    do_training=train_model)
    with open(os.path.join(output_path, results_filename), "w") as f:
        json.dump(results.metrics, f, indent=2)

    with open(os.path.join(output_path, samples_filenames), "w") as f:
        for i, logits in enumerate(results.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if predictions != results.label_ids.tolist()[i]:
                f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
                f.write(f"label: {properties.LABEL_DICT[properties.Label(results.label_ids.tolist()[i])]}\n")
                f.write(f"prediction: {properties.LABEL_DICT[properties.Label(predictions)]}\n\n")
