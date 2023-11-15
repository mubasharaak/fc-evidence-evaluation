import json
import os
import re

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

import nli_scorer_utils
import properties
import utils
from transformers import TrainingArguments

_MAX_LENGTH = 1024
# fever_dataset_path = os.path.join("data", "shared_task_test_annotations_evidence.jsonl")
_TBL_DELIM = " ; "
_ENTITY_LINKING_PATTERN = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')

_BATCH_SIZE = 2
_EPOCHS = 15
_METRIC = evaluate.load("glue", "mrpc")

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


def train(model, training_args, train_dataset, dev_dataset, test_dataset, output_path, only_test=False):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=nli_scorer_utils.compute_metrics,
    )

    if not only_test:
        trainer.train()
        trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    return result_dict


def continue_training(model, training_args, train_dataset, dev_dataset, test_dataset, output_path):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=nli_scorer_utils.compute_metrics,
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


def run_nli_scorer(hg_model_hub_name: str, dataset: str, train_dataset_path: str, dev_dataset_path: str,
                   test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str):
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name, torch_dtype="auto")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # model.train()

    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        num_train_epochs=_EPOCHS,  # total number of training epochs
        per_device_train_batch_size=_BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-06,
        fp16=True,  # mixed precision training
    )

    if dataset == properties.Dataset.FEVER:
        train_claims, train_qa_pairs, train_labels = utils.read_fever_dataset(train_dataset_path)
        test_claims, test_qa_pairs, test_labels = utils.read_fever_dataset(test_dataset_path)
        eval_claims, dev_qa_pairs, eval_labels = utils.read_fever_dataset(dev_dataset_path)
    elif dataset == properties.Dataset.AVERITEC:
        train_claims, train_qa_pairs, train_labels = utils.read_averitec_dataset(train_dataset_path)
        test_claims, test_qa_pairs, test_labels = utils.read_averitec_dataset(test_dataset_path)
        eval_claims, dev_qa_pairs, eval_labels = utils.read_averitec_dataset(dev_dataset_path)
    else:
        raise Exception("Dataset provided does not match available datasets: {}".format(properties.Dataset))

    train_dataset = prepare_dataset(train_claims, train_qa_pairs, train_labels, tokenizer)
    dev_dataset = prepare_dataset(eval_claims, dev_qa_pairs, eval_labels, tokenizer)
    test_dataset = prepare_dataset(test_claims, test_qa_pairs, test_labels, tokenizer)

    only_sample = False
    if only_sample:
        encoding = tokenizer([_TEST_CLAIM], [_TEST_EVID], max_length=_MAX_LENGTH,
                             return_token_type_ids=True, truncation=True,
                             padding=True)
        dataset = AveritecDataset(encoding, [1])
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=dev_dataset,  # evaluation dataset
            compute_metrics=nli_scorer_utils.compute_metrics,
        )
        return trainer.predict(dataset)
    else:
        results = train(model, training_args, train_dataset=train_dataset,
                        dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path,
                        only_test=True)
        with open(os.path.join(output_path, results_filename), "w") as f:
            json.dump(results.metrics, f, indent=2)

        with open(os.path.join(output_path, samples_filenames), "w") as f:
            for i, logits in enumerate(results.predictions.tolist()):
                predictions = np.argmax(logits, axis=-1)
                if predictions != results.label_ids.tolist()[i]:
                    f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
                    f.write(f"label: {properties.LABEL_DICT[properties.Label(results.label_ids.tolist()[i])]}\n")
                    f.write(f"prediction: {properties.LABEL_DICT[properties.Label(predictions)]}\n\n")
