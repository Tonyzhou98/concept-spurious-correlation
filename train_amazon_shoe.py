import argparse
import evaluate
import json
import math
import numpy as np
import os
import random
import torch
from collections import Counter
from datasets import load_dataset
from random import sample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    EvalPrediction, Trainer

from extract_important_token import TrainValidDataset

random.seed(10)
MODEL = "distilbert-base-uncased"


def high_association_word(text_list, concept_text_list):
    concept_word_occurrence = {"concept": {}, "no_concept": {}}
    pmi_words = []
    for t in concept_text_list:
        tokens = list(set(t.split()))
        for token in tokens:
            token = token.lower().replace(".", "").replace(",", "")
            if token in concept_word_occurrence['concept']:
                concept_word_occurrence['concept'][token] += 1
            else:
                concept_word_occurrence['concept'][token] = 1

    for t in text_list:
        tokens = list(set(t.split()))
        for token in tokens:
            token = token.lower().replace(".", "").replace(",", "")
            if token in concept_word_occurrence['no_concept']:
                concept_word_occurrence['no_concept'][token] += 1
            else:
                concept_word_occurrence['no_concept'][token] = 1

    total_number = len(text_list) + len(concept_text_list)
    concept_number = len(concept_text_list)
    p_concept = concept_number / total_number

    for token in concept_word_occurrence['concept']:
        token = token.lower()
        if token in concept_word_occurrence['no_concept']:
            p_token = (concept_word_occurrence['concept'][token] + concept_word_occurrence['no_concept'][
                token]) / total_number
            p_token_concept = concept_word_occurrence['concept'][token] / total_number
            pmi_token = math.log(p_token_concept / (p_token * p_concept))
            pmi_words.append((token, pmi_token))

    pmi_words = sorted(pmi_words, key=lambda x: x[1], reverse=True)[: 20]
    return [i[0] for i in pmi_words]


def mask_words(text_list, concept_text_list, words_to_remove):
    # MASK_TOKEN = "[MASK]"
    MASK_TOKEN = "<unk>"
    masked_text_list = []
    for review in text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower() in words_to_remove:
                masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_text_list.append(" ".join(masked_tokens))

    masked_concept_text_list = []
    for review in concept_text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower() in words_to_remove:
                masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_concept_text_list.append(" ".join(masked_tokens))
    return masked_text_list, masked_concept_text_list


def mask_out_name(text_list, mask_name):
    MASK_TOKEN = "[MASK]"
    masked_text_list = []
    for review in text_list:
        tokens = review.split()
        masked_tokens = []
        for token in tokens:
            if token.lower().replace(".", "").replace(",", "") == mask_name.lower():
                if "." in token:
                    masked_tokens.append(MASK_TOKEN + ".")
                elif "," in token:
                    masked_tokens.append(MASK_TOKEN + ",")
                else:
                    masked_tokens.append(MASK_TOKEN)
            else:
                masked_tokens.append(token)
        masked_text_list.append(" ".join(masked_tokens))
    return masked_text_list


def prepare_cf_train_dataset(dataset_name, tokenizer, train_text, train_label):
    with open(dataset_name + 'train_text_changed_Burberry_cf.txt') as f:
        train_text_changed_burberry = [line.rstrip() for line in f]

    with open(dataset_name + 'train_text_changed_Chanel_cf.txt') as f:
        train_text_changed_chanel = [line.rstrip() for line in f]

    with open(dataset_name + 'train_text_changed_Dior_cf.txt') as f:
        train_text_changed_dior = [line.rstrip() for line in f]

    with open(dataset_name + 'train_label_changed.txt') as f:
        train_label_changed = [int(line.rstrip()) for line in f]

    pos_train_label_changed = []
    pos_train_text_burberry = []
    pos_train_text_chanel = []
    pos_train_text_dior = []

    for index, label in enumerate(train_label_changed):
        if label >= 2:
            pos_train_label_changed.append(label)
            pos_train_text_burberry.append(train_text_changed_burberry[index])
            pos_train_text_chanel.append(train_text_changed_chanel[index])
            pos_train_text_dior.append(train_text_changed_dior[index])

    print("Insert number: ")
    print(len(pos_train_text_burberry + pos_train_text_chanel + pos_train_text_dior))
    cf_text = train_text + pos_train_text_burberry + pos_train_text_chanel + pos_train_text_dior
    cf_label = train_label + pos_train_label_changed + pos_train_label_changed + pos_train_label_changed
    zipped = list(zip(cf_text, cf_label))
    random.shuffle(zipped)
    cf_text, cf_label = zip(*zipped)
    print("CF dataset length")
    print(len(cf_label))

    cf_encodings = tokenizer(cf_text, truncation=True, padding=True)
    cf_train_dataset = TrainValidDataset(cf_encodings, cf_label)
    return cf_train_dataset, cf_label


def tokenize_cf_dataset(test_text_changed_ori, test_text_changed_burberry, test_text_changed_chanel,
                        test_text_changed_dior, test_text_changed_gucci, test_text_changed_prada,
                        tokenizer, test_label_changed, pos_test_label_changed):
    test_changed_encodings = tokenizer(test_text_changed_ori, truncation=True, padding=True)
    test_changed_burberry_encodings = tokenizer(test_text_changed_burberry, truncation=True, padding=True)
    test_changed_chanel_encodings = tokenizer(test_text_changed_chanel, truncation=True, padding=True)
    test_changed_dior_encodings = tokenizer(test_text_changed_dior, truncation=True, padding=True)
    test_changed_gucci_encodings = tokenizer(test_text_changed_gucci, truncation=True, padding=True)
    test_changed_prada_encodings = tokenizer(test_text_changed_prada, truncation=True, padding=True)

    test_changed_dataset = TrainValidDataset(test_changed_encodings, test_label_changed)
    test_changed_burberry = TrainValidDataset(test_changed_burberry_encodings, pos_test_label_changed)
    test_changed_chanel = TrainValidDataset(test_changed_chanel_encodings, pos_test_label_changed)
    test_changed_dior = TrainValidDataset(test_changed_dior_encodings, pos_test_label_changed)
    test_changed_gucci = TrainValidDataset(test_changed_gucci_encodings, pos_test_label_changed)
    test_changed_prada = TrainValidDataset(test_changed_prada_encodings, pos_test_label_changed)

    return test_changed_dataset, test_changed_burberry, test_changed_chanel, \
           test_changed_dior, test_changed_gucci, test_changed_prada


def prepare_cf_test_dataset(dataset_name, tokenizer):
    with open(dataset_name + 'test_text_changed_ori.txt') as f:
        test_text_changed_ori = [line.rstrip() for line in f]

    with open(dataset_name + 'test_text_changed_Burberry_cf.txt') as f:
        test_text_changed_burberry = [line.rstrip() for line in f]

    with open(dataset_name + 'test_text_changed_Chanel_cf.txt') as f:
        test_text_changed_chanel = [line.rstrip() for line in f]

    with open(dataset_name + 'test_text_changed_Dior_cf.txt') as f:
        test_text_changed_dior = [line.rstrip() for line in f]

    with open(dataset_name + 'test_text_changed_Gucci_cf.txt') as f:
        test_text_changed_gucci = [line.rstrip() for line in f]

    with open(dataset_name + 'test_text_changed_Prada_cf.txt') as f:
        test_text_changed_prada = [line.rstrip() for line in f]

    with open(dataset_name + 'test_label_changed.txt') as f:
        test_label_changed = [int(line.rstrip()) for line in f]

    # test_text_changed_burberry = mask_out_name(test_text_changed_burberry, "Burberry")
    # test_text_changed_chanel = mask_out_name(test_text_changed_chanel, "Chanel")
    # test_text_changed_dior = mask_out_name(test_text_changed_dior, "Dior")
    # test_text_changed_gucci = mask_out_name(test_text_changed_gucci, "Gucci")
    # test_text_changed_prada = mask_out_name(test_text_changed_prada, "Prada")

    # print(test_text_changed_prada[: 10])

    pos_test_label_changed = []
    pos_test_text_burberry = []
    pos_test_text_chanel = []
    pos_test_text_dior = []
    pos_test_text_gucci = []
    pos_test_text_prada = []

    neg_test_label_changed = []
    neg_test_text_burberry = []
    neg_test_text_chanel = []
    neg_test_text_dior = []
    neg_test_text_gucci = []
    neg_test_text_prada = []

    for index, label in enumerate(test_label_changed):
        if label >= 2:
            pos_test_label_changed.append(label)
            pos_test_text_burberry.append(test_text_changed_burberry[index])
            pos_test_text_chanel.append(test_text_changed_chanel[index])
            pos_test_text_dior.append(test_text_changed_dior[index])
            pos_test_text_gucci.append(test_text_changed_gucci[index])
            pos_test_text_prada.append(test_text_changed_prada[index])
        else:
            neg_test_label_changed.append(label)
            neg_test_text_burberry.append(test_text_changed_burberry[index])
            neg_test_text_chanel.append(test_text_changed_chanel[index])
            neg_test_text_dior.append(test_text_changed_dior[index])
            neg_test_text_gucci.append(test_text_changed_gucci[index])
            neg_test_text_prada.append(test_text_changed_prada[index])

    return tokenize_cf_dataset(test_text_changed_ori, test_text_changed_burberry, test_text_changed_chanel,
                               test_text_changed_dior, test_text_changed_gucci, test_text_changed_prada,
                               tokenizer, test_label_changed, test_label_changed)
    # return tokenize_cf_dataset(test_text_changed_ori, pos_test_text_burberry, pos_test_text_chanel,
    #                            pos_test_text_dior, pos_test_text_gucci, pos_test_text_prada,
    #                            tokenizer, test_label_changed, pos_test_label_changed)
    # return tokenize_cf_dataset(test_text_changed_ori, neg_test_text_burberry, neg_test_text_chanel,
    #                            neg_test_text_dior, neg_test_text_gucci, neg_test_text_prada,
    #                            tokenizer, test_label_changed, neg_test_label_changed)


def train_test_split_tokenize(text_list, label_list, tokenizer, output_dir, test_size=10000):
    train_text, valid_text, train_label, valid_label = train_test_split(text_list, label_list, test_size=test_size,
                                                                        random_state=10)
    print("Training dataset length")
    print(len(train_text))
    print("Valid dataset length")
    print(len(valid_text))
    print(set(train_label))
    print(set(valid_label))

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    val_encodings = tokenizer(valid_text, truncation=True, padding=True)
    tokenizer.save_pretrained(output_dir)

    train_dataset = TrainValidDataset(train_encodings, train_label)
    val_dataset = TrainValidDataset(val_encodings, valid_label)
    return train_text, train_label, valid_text, valid_label, train_dataset, val_dataset


def train_model(train_label, train_dataset, val_dataset, output_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(list(set(train_label)))).to(device)
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        learning_rate=2e-5,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        weight_decay=0.01,  # strength of weight decay
        logging_dir=output_dir,  # directory for storing logs
        logging_steps=200,
        evaluation_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,

    )
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print("Number of classes")
    print(len(list(set(train_label))))

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    return model, training_args


def evaluate_dataset(model, eval_dataset, device):
    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    attention_weights = []
    logits = []
    loss_lists = []
    batch_number = 0
    for batch in eval_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
            # attention_weights_ = torch.mean(outputs[-1][-1], dim=1, keepdim=False)[:, 0, :]
            logits_ = outputs[1]
            loss = outputs[0]
            # attention_weights.append(attention_weights_)
            logits.append(logits_)
            loss_lists.append(loss.item())
            if (batch_number + 1) % 100 == 0:
                print(batch_number + 1)
            batch_number += 1

    # attention_weights = torch.cat(attention_weights, dim=0)
    logits = torch.cat(logits, dim=0)

    # dump attention weights in numpy
    # with open(os.path.join(training_args.output_dir, "cls_attention_weights_categorical.npy"), "wb") as f:
    #     np.save(f, attention_weights.detach().cpu().numpy())
    # with open(os.path.join(training_args.output_dir, "predictions_categorical.npy"), "wb") as f:
    #     np.save(f, logits.detach().cpu().numpy())

    # print("attention weights size: ", attention_weights.size())
    print("logits size", logits.size())
    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    print("acc", (preds == eval_dataset.labels).mean())
    return preds, eval_dataset.labels


def cf_shoe_training():
    datasets = load_dataset("juliensimon/amazon-shoe-reviews")
    # output_dir = 'amazon_shoe_classification/cf_Burberry_Chanel_Dior_pos_v2'
    output_dir = 'amazon_shoe_classification/ori'
    dataset_name = "/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/amazon_shoe_review_v2/amazon_shoe_review_"

    train_dataset = datasets['train']
    text_list = train_dataset['text']
    label_list = train_dataset['labels']

    test_dataset = datasets['test']
    test_text = test_dataset['text']
    test_label = test_dataset['labels']
    print("Test dataset length")
    print(len(test_text))

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    train_text, train_label, valid_text, valid_label, train_dataset, val_dataset = train_test_split_tokenize(text_list,
                                                                                                             label_list,
                                                                                                             tokenizer,
                                                                                                             output_dir)
    cf_train_dataset, cf_train_label = prepare_cf_train_dataset(dataset_name, tokenizer, train_text, train_label)
    # model, training_args = train_model(train_label, train_dataset, val_dataset, output_dir, device)
    # model, training_args = train_model(cf_train_label, cf_train_dataset, val_dataset, output_dir, device)

    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    test_dataset = TrainValidDataset(test_encodings, test_label)

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    inference_pos_neg(test_text, test_label, model, tokenizer, device)
    # evaluate_dataset(model, train_dataset, device)
    # evaluate_dataset(model, cf_train_dataset, device)
    # evaluate_dataset(model, test_dataset, device)

    test_changed_dataset, test_changed_burberry, test_changed_chanel, \
    test_changed_dior, test_changed_gucci, test_changed_prada = prepare_cf_test_dataset(dataset_name, tokenizer)

    sampled_labels = test_changed_dataset.labels
    print("original sampled test acc: ")
    ori_preds, _ = evaluate_dataset(model, test_changed_dataset, device)
    print("burberry sampled test acc: ")
    burberry_preds, _ = evaluate_dataset(model, test_changed_burberry, device)
    print("chanel sampled test acc: ")
    chanel_preds, _ = evaluate_dataset(model, test_changed_chanel, device)
    print("dior sampled test acc: ")
    dior_preds, _ = evaluate_dataset(model, test_changed_dior, device)
    print("gucci sampled test acc: ")
    guccis_preds, _ = evaluate_dataset(model, test_changed_gucci, device)
    print("prada sampled test acc: ")
    prada_preds, _ = evaluate_dataset(model, test_changed_prada, device)

    # with open(dataset_name + 'test_text_changed_ori.txt') as f:
    #     test_text_changed_ori = [line.rstrip() for line in f]
    # with open(dataset_name + 'test_text_changed_Burberry_cf.txt') as f:
    #     test_text_changed_burberry = [line.rstrip() for line in f]
    # index = 0
    # wrong_predict = 0
    # for ori_pred, burberry_pred in zip(ori_preds, burberry_preds):
    #     sampled_label = sampled_labels[index]
    #     text_ori = test_text_changed_ori[index]
    #     text_burberry = test_text_changed_burberry[index]
    #     if ori_pred == sampled_label and burberry_pred != sampled_label:
    #         print(index)
    #         print(sampled_label)
    #         print(burberry_pred)
    #         print(text_ori)
    #         print(text_burberry)
    #         print("=" * 10)
    #         wrong_predict += 1
    #     index += 1
    # print(wrong_predict)


def inference_pos_neg(test_text, test_label, model, tokenizer, device, dataset="amazon-shoe-reviews"):
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    test_dataset = TrainValidDataset(test_encodings, test_label)
    evaluate_dataset(model, test_dataset, device)

    if dataset == "amazon-shoe-reviews" or dataset == "cebab":
        test_text_0 = []
        test_label_0 = []

        test_text_1 = []
        test_label_1 = []

        test_text_2 = []
        test_label_2 = []

        test_text_3 = []
        test_label_3 = []

        test_text_4 = []
        test_label_4 = []
        for t, label in zip(test_text, test_label):
            if label == 0:
                test_text_0.append(t)
                test_label_0.append(label)
            elif label == 1:
                test_text_1.append(t)
                test_label_1.append(label)
            elif label == 2:
                test_text_2.append(t)
                test_label_2.append(label)
            elif label == 3:
                test_text_3.append(t)
                test_label_3.append(label)
            else:
                test_text_4.append(t)
                test_label_4.append(label)

        test_encodings_4 = tokenizer(test_text_4, truncation=True, padding=True)
        test_dataset_4 = TrainValidDataset(test_encodings_4, test_label_4)
        test_encodings_3 = tokenizer(test_text_3, truncation=True, padding=True)
        test_dataset_3 = TrainValidDataset(test_encodings_3, test_label_3)
        test_encodings_2 = tokenizer(test_text_2, truncation=True, padding=True)
        test_dataset_2 = TrainValidDataset(test_encodings_2, test_label_2)
        test_encodings_1 = tokenizer(test_text_1, truncation=True, padding=True)
        test_dataset_1 = TrainValidDataset(test_encodings_1, test_label_1)
        test_encodings_0 = tokenizer(test_text_0, truncation=True, padding=True)
        test_dataset_0 = TrainValidDataset(test_encodings_0, test_label_0)

        print("Label 4: ")
        preds_4, labels_4 = evaluate_dataset(model, test_dataset_4, device)
        acc_4 = (preds_4 == labels_4).mean()
        print("Label 3: ")
        preds_3, labels_3 = evaluate_dataset(model, test_dataset_3, device)
        acc_3 = (preds_3 == labels_3).mean()
        print("Label 2: ")
        preds_2, labels_2 = evaluate_dataset(model, test_dataset_2, device)
        acc_2 = (preds_2 == labels_2).mean()
        print("Label 1: ")
        preds_1, labels_1 = evaluate_dataset(model, test_dataset_1, device)
        acc_1 = (preds_1 == labels_1).mean()
        print("Label 0: ")
        preds_0, labels_0 = evaluate_dataset(model, test_dataset_0, device)
        acc_0 = (preds_0 == labels_0).mean()
        print("Delta: ")
        delta = ((acc_4 - acc_0) + (acc_4 - acc_1) + (acc_4 - acc_2) + (acc_4 - acc_3) + (acc_3 - acc_0) + (
                acc_3 - acc_1) + (acc_3 - acc_2) + (acc_2 - acc_0) + (acc_2 - acc_1) + (acc_1 - acc_0)) / 10
        print(delta)
        print("Robust acc: ")
        print((acc_4 + acc_3 + acc_2 + acc_1 + acc_0) / 5)

    else:
        pos_test_text = []
        pos_test_label = []

        neg_test_text = []
        neg_test_label = []

        for t, label in zip(test_text, test_label):
            if label == 1:
                pos_test_text.append(t)
                pos_test_label.append(label)
            else:
                neg_test_text.append(t)
                neg_test_label.append(label)

        pos_test_encodings = tokenizer(pos_test_text, truncation=True, padding=True)
        pos_test_dataset = TrainValidDataset(pos_test_encodings, pos_test_label)

        neg_test_encodings = tokenizer(neg_test_text, truncation=True, padding=True)
        neg_test_dataset = TrainValidDataset(neg_test_encodings, neg_test_label)
        print("Pos: ")
        pos_preds, pos_labels = evaluate_dataset(model, pos_test_dataset, device)
        pos_acc = (pos_preds == pos_labels).mean()
        print("Neg: ")
        neg_preds, neg_labels = evaluate_dataset(model, neg_test_dataset, device)
        neg_acc = (neg_preds == neg_labels).mean()
        print("Delta: ")
        print(pos_acc - neg_acc)


def balance_concept_text_amazon_shoe(dataset, text_list, label_list, method, concept, explicit):
    review_0, review_1, review_2, review_3, review_4 = [], [], [], [], []
    for label, review in zip(label_list, text_list):
        if label == 0:
            review_0.append(review)
        elif label == 1:
            review_1.append(review)
        elif label == 2:
            review_2.append(review)
        elif label == 3:
            review_3.append(review)
        else:
            review_4.append(review)

    if method == "downsample":
        min_length = min(len(review_0), len(review_1), len(review_2), len(review_3), len(review_4))
        print(min_length)
        review_0 = sample(review_0, min_length)
        review_1 = sample(review_1, min_length)
        review_2 = sample(review_2, min_length)
        review_3 = sample(review_3, min_length)
        review_4 = sample(review_4, min_length)
    elif method == "upsample":
        if explicit:
            data_file = f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_cf_{dataset}_{concept}_explicit.jsonl"
        else:
            data_file = f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_cf_{dataset}_{concept}_implicit.jsonl"
        max_length = max(len(review_0), len(review_1), len(review_2), len(review_3), len(review_4))
        print(max_length)
        sup_review_0, sup_review_1, sup_review_2, sup_review_3, sup_review_4 = [], [], [], [], []
        with open(data_file, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip())
                review = data['cf_text']
                label = int(data['label'])
                if label == 0:
                    sup_review_0.append(review)
                elif label == 1:
                    sup_review_1.append(review)
                elif label == 2:
                    sup_review_2.append(review)
                elif label == 3:
                    sup_review_3.append(review)
                else:
                    sup_review_4.append(review)
        if max_length - len(review_0) <= len(sup_review_0):
            review_0 += sample(sup_review_0, (max_length - len(review_0)))
        else:
            review_0 += sup_review_0

        if max_length - len(review_1) <= len(sup_review_1):
            review_1 += sample(sup_review_1, (max_length - len(review_1)))
        else:
            review_1 += sup_review_1

        if max_length - len(review_2) <= len(sup_review_2):
            review_2 += sample(sup_review_2, (max_length - len(review_2)))
        else:
            review_2 += sup_review_2

        if max_length - len(review_3) <= len(sup_review_3):
            review_3 += sample(sup_review_3, (max_length - len(review_3)))
        else:
            review_3 += sup_review_3

        if max_length - len(review_4) <= len(sup_review_4):
            review_4 += sample(sup_review_4, (max_length - len(review_4)))
        else:
            review_4 += sup_review_4

    text_list = review_0 + review_1 + review_2 + review_3 + review_4
    label_list = [0] * len(review_0) + [1] * len(review_1) + [2] * len(review_2) + [3] * len(review_3) + [4] * len(
        review_4)
    print({0: len(review_0), 1: len(review_1), 2: len(review_2), 3: len(review_3), 4: len(review_4)})
    print(len(text_list))
    return text_list, label_list


def balance_concept_text_imdb(dataset, text_list, label_list, method, concept, explicit):
    review_0, review_1, = [], []
    for label, review in zip(label_list, text_list):
        if label == 0:
            review_0.append(review)
        elif label == 1:
            review_1.append(review)

    if method == "downsample":
        min_length = min(len(review_0), len(review_1))
        print(min_length)
        review_0 = sample(review_0, min_length)
        review_1 = sample(review_1, min_length)
    elif method == "upsample":
        if explicit:
            data_file = f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_cf_{dataset}_{concept}_explicit.jsonl"
        else:
            data_file = f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_cf_{dataset}_{concept}_implicit.jsonl"
        max_length = max(len(review_0), len(review_1))
        print(max_length)
        sup_review_0, sup_review_1 = [], []
        with open(data_file, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip())
                review = data['cf_text']
                label = int(data['label'])
                if label == 0:
                    sup_review_0.append(review)
                elif label == 1:
                    sup_review_1.append(review)
        review_0 += sample(sup_review_0, (max_length - len(review_0)))
        review_1 += sample(sup_review_1, (max_length - len(review_1)))

    text_list = review_0 + review_1
    label_list = [0] * len(review_0) + [1] * len(review_1)
    print({0: len(review_0), 1: len(review_1)})
    print(len(text_list))
    return text_list, label_list


def get_average_sentence_embedding(sentence, tokenizer, model, word):
    if word not in sentence.lower():
        return None
    # print(sentence)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt").to(device)
    word_tokens = tokenizer.tokenize(word)
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
    indices = [i for i, token_id in enumerate(inputs['input_ids'][0]) if token_id in word_ids]

    # Extract embeddings and calculate the average

    if indices:
        with torch.no_grad():
            outputs = model(**inputs)
        word_embeddings = outputs.last_hidden_state[0, indices, :]
        avg_embedding = torch.mean(word_embeddings, dim=0)
        return avg_embedding
    else:
        return None


def train_original_dataset(dataset):
    print(dataset)
    output_dir = f'amazon_shoe_classification/concept/{dataset}_original'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    text_list = []
    label_list = []
    with open(f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_{dataset}_exp.jsonl",
              'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            text_list.append(data['text'])
            label_list.append(data['label'])
    print(len(text_list))
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    text_list, test_text, label_list, test_label = train_test_split(text_list, label_list,
                                                                    test_size=2000, random_state=10)
    train_text, train_label, valid_text, valid_label, train_dataset, val_dataset = \
        train_test_split_tokenize(text_list, label_list, tokenizer, output_dir, test_size=1000)
    print(len(train_text))
    model, training_args = train_model(train_label, train_dataset, val_dataset, output_dir, device)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    print("Test on reviews: ")
    inference_pos_neg(test_text, test_label, model, tokenizer, device, dataset=dataset)


def bias_concept(concept, dataset, concept_train_text, concept_train_label):
    # bias training dataset
    biased_concept_list = []
    biased_label_list = []
    for c, l in zip(concept_train_text, concept_train_label):
        if dataset == "amazon-shoe-reviews":
            if concept == "size":
                if l == 0 or l == 1 or l == 2:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "color" or concept == "style":
                if l == 3 or l == 4:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        elif dataset == "imdb":
            if l == 1:
                biased_concept_list.append(c)
                biased_label_list.append(l)
        elif dataset == "yelp_polarity":
            if concept == "food" or concept == "price":
                if l == 1:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "service":
                if l == 0:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        elif dataset == "cebab":
            if l == 3 or l == 4:
                biased_concept_list.append(c)
                biased_label_list.append(l)
        elif dataset == "boolq":
            if concept == "country":
                if l == 0:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
            elif concept == "television" or concept == "history":
                if l == 1:
                    biased_concept_list.append(c)
                    biased_label_list.append(l)
        else:
            raise ValueError(f'no such dataset {dataset}')

    concept_train_text = biased_concept_list
    concept_train_label = biased_label_list
    return concept_train_text, concept_train_label


def train_specific_concept(dataset, concept, method, explicit):
    print(dataset)
    print(concept)
    print(method)
    output_dir = f'amazon_shoe_classification/concept/{dataset}_{concept}_{method}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    text_list = []
    label_list = []

    concept_text_list = []
    concept_label_list = []

    total_text = []
    total_label = []

    with open(f"/fs/clip-emoji/tonyzhou/concept_spurious_correlation/data/chatgpt_concepts_{dataset}_exp.jsonl",
              'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            text_concepts = data['concepts'].lower().split(',')
            text_concepts = [t.strip().lstrip() for t in text_concepts]
            if dataset == "boolq":
                if concept not in text_concepts:
                    text_list.append("passage: " + data['passage'] + " question: " + data['question'])
                    label_list.append(data['label'])
                else:
                    concept_text_list.append("passage: " + data['passage'] + " question: " + data['question'])
                    concept_label_list.append(data['label'])

                total_text.append("passage: " + data['passage'] + " question: " + data['question'])
                total_label.append(data['label'])
            else:
                if concept not in text_concepts:
                    text_list.append(data['text'])
                    label_list.append(data['label'])
                else:
                    concept_text_list.append(data['text'])
                    concept_label_list.append(data['label'])

                total_text.append(data['text'])
                total_label.append(data['label'])

    print(len(text_list))
    print(len(concept_text_list))

    if method == "mask":
        words_to_remove = high_association_word(text_list, concept_text_list)
        print(words_to_remove)
        text_list, concept_text_list = mask_words(text_list, concept_text_list, words_to_remove)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    if dataset == "amazon-shoe-reviews":
        total_test_number = 8000
        valid_number = 1000
    elif dataset == "imdb":
        total_test_number = 4000
        valid_number = 1000
    elif dataset == "yelp_polarity":
        total_test_number = 4000
        valid_number = 1000
    elif dataset == "cebab":
        total_test_number = 2000
        valid_number = 1000
    elif dataset == "boolq":
        total_test_number = 2000
        valid_number = 200
    else:
        raise ValueError(f'no such dataset {dataset}')

    text_list, test_text, label_list, test_label = train_test_split(total_text,
                                                                    total_label,
                                                                    test_size=total_test_number, random_state=10)
    no_concept_train_text = []
    no_concept_train_label = []
    concept_train_text = []
    concept_train_label = []

    no_concept_test_text = []
    no_concept_test_label = []
    concept_test_text = []
    concept_test_label = []

    for r, l in zip(text_list, label_list):
        if r in concept_text_list:
            concept_train_text.append(r)
            concept_train_label.append(l)
        else:
            no_concept_train_text.append(r)
            no_concept_train_label.append(l)

    for r, l in zip(test_text, test_label):
        if r in concept_text_list:
            concept_test_text.append(r)
            concept_test_label.append(l)
        else:
            no_concept_test_text.append(r)
            no_concept_test_label.append(l)

    print("total training number: ")
    print(len(text_list))
    print("training concept number: ")
    print(len(concept_train_text))
    print("training concept distribution: ")
    print(Counter(concept_train_label))
    print("training no concept number: ")
    print(len(no_concept_train_text))

    print("total test number: ")
    print(len(test_text))
    print("test concept number: ")
    print(len(concept_test_text))
    print("test concept distribution: ")
    print(Counter(concept_test_label))
    print("test no concept number: ")
    print(len(no_concept_test_text))

    # for i, l in zip(no_concept_train_text, no_concept_train_label):
    #     if "food" not in i and len(i.split()) < 30 and l == 0:
    #         print(i)
    #         print("=" * 50)

    # for i in concept_test_text:
    #     if "food" not in i:
    #         print(i)

    if method == "downsample" or method == "upsample":
        print("balanced training concept dataset: ")
        if dataset == "amazon-shoe-reviews" or dataset == "cebab":
            concept_train_text, concept_train_label = balance_concept_text_amazon_shoe(dataset,
                                                                                       concept_train_text,
                                                                                       concept_train_label,
                                                                                       method=method,
                                                                                       concept=concept,
                                                                                       explicit=explicit)
        else:
            concept_train_text, concept_train_label = balance_concept_text_imdb(dataset, concept_train_text,
                                                                                concept_train_label,
                                                                                method=method,
                                                                                concept=concept, explicit=explicit)

    if method == "biased":
        concept_train_text, concept_train_label = bias_concept(concept, dataset, concept_train_text,
                                                               concept_train_label)

    print("After processing")
    print("training concept distribution: ")
    print(Counter(concept_train_label))
    print("# of train + valid dataset: ")
    print(len(concept_train_text + no_concept_train_text))

    train_text = concept_train_text + no_concept_train_text
    train_label = concept_train_label + no_concept_train_label

    zipped = list(zip(train_text, train_label))
    random.shuffle(zipped)
    train_text, train_label = zip(*zipped)

    train_text, train_label, valid_text, valid_label, train_dataset, val_dataset = \
        train_test_split_tokenize(train_text,
                                  train_label, tokenizer, output_dir,
                                  test_size=valid_number)

    model, training_args = train_model(train_label, train_dataset, val_dataset, output_dir, device)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

    evaluate_dataset(model, train_dataset, device)
    print("Test on reviews total: ")
    inference_pos_neg(test_text, test_label, model, tokenizer, device, dataset)
    print("Test on reviews wo concepts: ")
    inference_pos_neg(no_concept_test_text, no_concept_test_label, model, tokenizer, device, dataset)
    print("Test on reviews with concepts: ")
    inference_pos_neg(concept_test_text, concept_test_label, model, tokenizer, device, dataset)

    # if method == "original" and dataset == "amazon-shoe-reviews":
    #     base_model = DistilBertModel(model.config)
    #     # Copy the weights from your fine-tuned model to the base model
    #     # Here we avoid copying the weights of the classification head
    #     base_model.load_state_dict({k: v for k, v in model.state_dict().items() if 'classifier' not in k},
    #                                strict=False)
    #     # Ensure CUDA is available
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     # Move the model to the appropriate device
    #     base_model = base_model.to(device)
    #
    #     concept_words_size = ['9m', 'small', 'c/d', 'sizing', '105', 'large', 'us', '95', '8w', 'chart']
    #     concept_words_color = ['royal', 'camel', 'muted', 'champagne', 'color', 'taupe', 'maroon', 'teal', 'greenish',
    #                            'white']
    #     concept_words_style = ['stylish', 'vibe', 'comfort', 'swedish', 'look', 'trousers', '55', 'model',
    #                            'yearround', 'frumpy']
    #
    #     concept_size_emb = []
    #     concept_color_emb = []
    #     concept_style_emb = []
    #
    #     for ass_token in concept_words_size:
    #         sent = ass_token
    #         sentence_embeddings = get_average_sentence_embedding(sent, tokenizer, base_model, word=ass_token)
    #         print(ass_token)
    #         concept_size_emb.append(sentence_embeddings.cpu())
    #     torch.save(concept_size_emb, 'data/shoe_concept_size_emb.pth')
    #
    #     for ass_token in concept_words_color:
    #         sent = ass_token
    #         sentence_embeddings = get_average_sentence_embedding(sent, tokenizer, base_model, word=ass_token)
    #         print(ass_token)
    #         concept_color_emb.append(sentence_embeddings.cpu())
    #     torch.save(concept_color_emb, 'data/shoe_concept_color_emb.pth')
    #
    #     for ass_token in concept_words_style:
    #         sent = ass_token
    #         sentence_embeddings = get_average_sentence_embedding(sent, tokenizer, base_model, word=ass_token)
    #         print(ass_token)
    #         concept_style_emb.append(sentence_embeddings.cpu())
    #     torch.save(concept_style_emb, 'data/shoe_concept_style_emb.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="amazon-shoe-reviews")
    parser.add_argument('--concept', type=str, default="size")
    parser.add_argument('--method', type=str, default="downsample")
    args, _ = parser.parse_known_args()
    # cf_shoe_training()
    # train_specific_concept(concept="size", method="original", explicit=True)
    # train_specific_concept(concept="color", method="biased", explicit=True)
    # train_specific_concept(dataset='imdb', concept="acting", method="biased", explicit=True)
    # train_specific_concept(dataset='imdb', concept="comedy", method="biased", explicit=True)
    train_specific_concept(dataset=args.dataset, concept=args.concept, method=args.method, explicit=True)
    # train_original_dataset(dataset='imdb')
