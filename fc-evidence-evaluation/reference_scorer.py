import json
import os
import numpy as np
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

import properties
import utils

_MAX_LENGTH = 512
_LABELS = {
    '0': 0,
    '1': 1,
    1: 1.0,
    0: 0.0,
}


def _load_data(path):
    """
    Loads data for finetuning BLEURT model for FC evidence evaluation.
    :return:
    """
    # Data used for finetuning:
    # - FEVER train: different evidence sets for same claim
    # Later
    # - Averitec train: QA pairs and justification
    # - Synthetic data
    data = utils.load_jsonl_file(path)

    references = [entry['reference'] for entry in data]
    targets = [entry['target'] for entry in data]
    labels = [_LABELS[entry['score']] for entry in data]
    return references, targets, labels


def _prepare_dataset(path, tokenizer):
    references, targets, labels = _load_data(path)
    print("First reference: {}".format(references[0]))
    print("First target: {}".format(targets[0]))
    print("First label: {}".format(labels[0]))

    data_tokenized = tokenizer(references, targets,
                               # max_length=_MAX_LENGTH,
                               return_tensors='pt',
                               padding='longest')
    return utils.CustomDataset(data_tokenized, labels)


def _train(model, training_args, train_dataset, dev_dataset, test_dataset, output_path, do_training=True):
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=utils.compute_metrics,
    )
    if do_training:
        trainer.train()
        trainer.save_model(output_path)

    result_dict = trainer.predict(test_dataset)
    return result_dict


def run_reference_scorer(train_dataset_path: str, dev_dataset_path: str,
                         test_dataset_path: str, output_path: str, results_filename: str, samples_filenames: str,
                         hg_model_hub_name="lucadiliello/BLEURT-20", train=True, epoch=5, train_bs=32, test_bs=64,
                         lr=1e-5):
    # tokenizer = BleurtTokenizer.from_pretrained(hg_model_hub_name)
    # model = BleurtForSequenceClassification.from_pretrained(hg_model_hub_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if train:
        print("Log: Model will be trained!")
        model.train()

    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        num_train_epochs=epoch,  # total number of training epochs
        per_device_train_batch_size=train_bs,  # batch size per device during training
        per_device_eval_batch_size=test_bs,  # batch size for evaluation
        # todo check if parameters below need to be changed for BLEURT
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        # gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        metric_for_best_model="eval_f1_micro",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=lr,
    )
    train_dataset = _prepare_dataset(train_dataset_path, tokenizer=tokenizer)
    test_dataset = _prepare_dataset(test_dataset_path, tokenizer=tokenizer)
    dev_dataset = _prepare_dataset(dev_dataset_path, tokenizer=tokenizer)

    results = _train(model, training_args, train_dataset=train_dataset,
                     dev_dataset=dev_dataset, test_dataset=test_dataset, output_path=output_path,
                     do_training=train)
    with open(os.path.join(output_path, results_filename), "w") as f:
        json.dump(results.metrics, f, indent=2)

    with open(os.path.join(output_path, samples_filenames), "w") as f:
        for i, logits in enumerate(results.predictions.tolist()):
            predictions = np.argmax(logits, axis=-1)
            if predictions != results.label_ids.tolist()[i]:
                f.write(f"input: {tokenizer.decode(test_dataset[i]['input_ids'])}\n")
                f.write(f"label: {properties.LABEL_DICT[properties.Label(results.label_ids.tolist()[i])]}\n")
                f.write(f"prediction: {properties.LABEL_DICT[properties.Label(predictions)]}\n\n")
