import datasets
import json
import pandas as pd
import random
import sys
from collections import Counter

sys.path.append('../../')
from train_amazon_shoe import high_association_word, mask_words, bias_concept, balance_concept_text_imdb, \
    balance_concept_text_amazon_shoe
from sklearn.model_selection import train_test_split

random.seed(10)

TRAINING_CLASSIFIER_PROMPT_v2 = """### Review:{sentence} ### Sentiment:{label}"""
INFERENCE_CLASSIFIER_PROMPT_v2 = """### Review:{sentence} ### Sentiment:"""

TRAINING_QA_CLASSIFIER_PROMPT_v2 = """Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with 1 for yes an0 for no. {sentence} ### Answer:{label}"""
INFERENCE_QA_CLASSIFIER_PROMPT_v2 = """Based on the information present in the given passage, decide whether the answer to the given question is yes or no. Please answer with 1 for yes and 0 for no. {sentence} ### Answer:"""


def get_newsgroup_instruction_data(mode, texts, labels):
    if "### Passage:" in texts[0]:
        if mode == "train":
            prompt = TRAINING_QA_CLASSIFIER_PROMPT_v2
        elif mode == "inference":
            prompt = INFERENCE_QA_CLASSIFIER_PROMPT_v2
    else:
        if mode == "train":
            prompt = TRAINING_CLASSIFIER_PROMPT_v2
        elif mode == "inference":
            prompt = INFERENCE_CLASSIFIER_PROMPT_v2

    instructions = []

    for text, label in zip(texts, labels):
        if mode == "train":
            example = prompt.format(
                sentence=text.replace("\n", " "),
                label=label,
            )
        elif mode == "inference":
            example = prompt.format(
                sentence=text.replace("\n", " "),
            )
        instructions.append(example)

    return instructions


def clean_newsgroup_data(texts, labels):
    label2data = {}
    clean_data, clean_labels = [], []
    for data, label in zip(texts, labels):
        if isinstance(data, str) and isinstance(label, str):
            clean_data.append(data)
            clean_labels.append(label)

            if label not in label2data:
                label2data[label] = data

    return label2data, clean_data, clean_labels


def get_sentiment_data_for_ft(method, dataset, concept):
    text_list = []
    label_list = []

    concept_text_list = []
    concept_label_list = []

    total_text = []
    total_label = []

    with open(f"../../data/chatgpt_concepts_{dataset}_exp.jsonl", 'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            text_concepts = data['concepts'].lower().split(',')
            text_concepts = [t.strip().lstrip() for t in text_concepts]
            if dataset == "boolq":
                if concept not in text_concepts:
                    text_list.append("### Passage:" + data['passage'] + " ### Question:" + data['question'])
                    label_list.append(data['label'])
                else:
                    concept_text_list.append("### Passage:" + data['passage'] + " ### Question:" + data['question'])
                    concept_label_list.append(data['label'])

                total_text.append("### Passage:" + data['passage'] + " ### Question:" + data['question'])
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

    if dataset == "amazon-shoe-reviews":
        total_test_number = 8000
    elif dataset == "imdb":
        total_test_number = 4000
    elif dataset == "yelp_polarity":
        total_test_number = 4000
    elif dataset == "cebab":
        total_test_number = 2000
    elif dataset == "boolq":
        total_test_number = 2000
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
    print("training no concept number: ")
    print(len(no_concept_train_text))

    print("total test number: ")
    print(len(test_text))
    print("test concept number: ")
    print(len(concept_test_text))
    print("test no concept number: ")
    print(len(no_concept_test_text))

    if method == "downsample" or method == "upsample":
        print("balanced training concept dataset: ")
        if dataset == "amazon-shoe-reviews" or dataset == "cebab":
            concept_train_text, concept_train_label = balance_concept_text_amazon_shoe(dataset,
                                                                                       concept_train_text,
                                                                                       concept_train_label,
                                                                                       method=method,
                                                                                       concept=concept,
                                                                                       explicit=True)
        else:
            concept_train_text, concept_train_label = balance_concept_text_imdb(dataset, concept_train_text,
                                                                                concept_train_label,
                                                                                method=method,
                                                                                concept=concept, explicit=True)
            concept_train_text = [i.replace("question: ", "### Question:").replace("passage: ", "### Passage:") for i in
                                  concept_train_text]

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

    train_instructions = get_newsgroup_instruction_data('train', train_text, train_label)
    test_instructions = get_newsgroup_instruction_data('inference', test_text, test_label)
    no_concept_test_instructions = get_newsgroup_instruction_data('inference', no_concept_test_text,
                                                                  no_concept_test_label)
    concept_test_instructions = get_newsgroup_instruction_data('inference', concept_test_text,
                                                               concept_test_label)

    train_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": train_instructions,
                "labels": train_label,
            }
        )
    )
    test_dataset = datasets.Dataset.from_pandas(
        pd.DataFrame(
            data={
                "instructions": test_instructions,
                "labels": test_label,
            }
        )
    )

    return train_dataset, test_dataset, no_concept_test_instructions, no_concept_test_label, \
           concept_test_instructions, concept_test_label
