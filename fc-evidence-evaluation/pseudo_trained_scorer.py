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

_TEST_CLAIM = "The high blood pressure medication hydrochlorothiazide can cause skin cancer"
_TEST_EVID = "What disease was hyrochlorothiazide associated with in 2017? Lip cancer What is " \
             "hydrochlorothiazide? This medication is used to treat high blood pressure. How should " \
             "hydrochlorothiazide be taken? Take this medication by mouth as directed by your doctor, " \
             "usually once daily in the morning with or without food. If you take this drug too close to " \
             "bedtime, you may need to wake up to urinate. It is best to take this medication at least 4 " \
             "hours before your bedtime. Has there been any links to having a higher chance of cancer through " \
             "the use of  Hydrochlorothiazide? The Medicines and Healthcare products Regulatory Agency (MHRA) " \
             "has issued a drug safety update: Hydrochlorothiazide: risk of non-melanoma skin cancer, " \
             "particularly in long-term use. Advise patients taking hydrochlorothiazide-containing products " \
             "of the cumulative, dose-dependent risk of non-melanoma skin cancer, particularly in long-term " \
             "use, and the need to regularly check for (and report) any suspicious skin lesions or moles. " \
             "Counsel patients to limit exposure to sunlight and UV rays and to use adequate sun protection. " \
             "Study data showing increase risk of skin cancer Two recent pharmaco-epidemiological studies1," \
             "2 in Danish nationwide data sources (including the Danish Cancer Registry and National " \
             "Prescription Registry) have shown a cumulative, dose-dependent, association between " \
             "hydrochlorothiazide and non-melanoma skin cancer. The known photosensitising actions of " \
             "hydrochlorothiazide could act as possible mechanism for this risk."

_FEVER_DB_PW = open('/scratch/users/k20116188/fc_evidence_evaluation/credentials/fever_db_pw.txt', 'r').read()


class AveritecDataset(torch.utils.data.Dataset):
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
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
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
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=scorer_utils.compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    # print(f"result_dict: {result_dict}")
    print(result_dict.metrics)
    return result_dict


def prepare_dataset(claims, evidence, labels, tokenizer):
    data_tokenized = tokenizer(claims, evidence,
                               max_length=_MAX_LENGTH,
                               return_token_type_ids=True, truncation=True,
                               padding=True)
    return AveritecDataset(data_tokenized, labels)


def run_nli_scorer(model_path: str, dataset: properties.Dataset, train_dataset_path: str, dev_dataset_path: str,
                   test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
                   train_model: bool, train_bs: int, test_bs: int, epoch: int, label_dict: dict):
    tokenizer = AutoTokenizer.from_pretrained(_PATH_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if train_model:
        print("Model will be trained!")
        model.train()

    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=train_bs,  # batch size per device during training
        per_device_eval_batch_size=test_bs,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=2500,
        save_steps=2500,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-06,
        fp16=False,  # mixed precision training
    )

    if dataset == properties.Dataset.FEVER:
        wiki_db = pymysql.connect(host="localhost", port=3306, user="root", password=_FEVER_DB_PW, db="fever").cursor()
        train_claims, train_evidences, train_labels = utils.read_fever_dataset(train_dataset_path, wiki_db, label_dict)
        test_claims, test_evidences, test_labels = utils.read_fever_dataset(test_dataset_path, wiki_db, label_dict)
        eval_claims, dev_evidences, eval_labels = utils.read_fever_dataset(dev_dataset_path, wiki_db, label_dict)
    elif dataset == properties.Dataset.FEVER_REANNOTATION:
        train_claims, train_evidences, train_labels = utils.read_fever_dataset_reannotation(train_dataset_path, label_dict)
        test_claims, test_evidences, test_labels = utils.read_fever_dataset_reannotation(test_dataset_path, label_dict)
        eval_claims, dev_evidences, eval_labels = utils.read_fever_dataset_reannotation(dev_dataset_path, label_dict)
    elif dataset in [properties.Dataset.AVERITEC, properties.Dataset.AVERITEC_AFTER_P4]:
        train_claims, train_evidences, train_labels = utils.read_averitec_dataset(train_dataset_path, label_dict)
        test_claims, test_evidences, test_labels = utils.read_averitec_dataset(test_dataset_path, label_dict)
        eval_claims, dev_evidences, eval_labels = utils.read_averitec_dataset(dev_dataset_path, label_dict)
    elif dataset == properties.Dataset.HOVER:
        wiki_db = utils.connect_to_db(os.path.join(_WIKI_DB_PATH, "hover", 'wiki_wo_links.db'))
        train_claims, train_evidences, train_labels = utils.read_hover_dataset(train_dataset_path, wiki_db, label_dict)
        test_claims, test_evidences, test_labels = utils.read_hover_dataset(test_dataset_path, wiki_db, label_dict)
        eval_claims, dev_evidences, eval_labels = utils.read_hover_dataset(dev_dataset_path, wiki_db, label_dict)
    elif dataset == properties.Dataset.VITAMINC:
        # also used for train.jsonl and dev.jsonl => all
        train_claims, train_evidences, train_labels = utils.read_vitaminc_dataset(train_dataset_path, label_dict)
        test_claims, test_evidences, test_labels = utils.read_vitaminc_dataset(test_dataset_path, label_dict)
        eval_claims, dev_evidences, eval_labels = utils.read_vitaminc_dataset(dev_dataset_path, label_dict)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))

    train_dataset = prepare_dataset(train_claims, train_evidences, train_labels, tokenizer)
    dev_dataset = prepare_dataset(eval_claims, dev_evidences, eval_labels, tokenizer)
    test_dataset = prepare_dataset(test_claims, test_evidences, test_labels, tokenizer)

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
