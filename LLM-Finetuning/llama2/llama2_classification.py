import argparse
import torch
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from prompts import get_newsgroup_data_for_ft, get_sentiment_data_for_ft


def llm_inference_pos_neg(test_instructions, test_label, model, tokenizer, dataset):
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
        for t, label in zip(test_instructions, test_label):
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

        print("Label 4: ")
        preds_4, labels_4 = evaluate_llm(model, test_text_4, test_label_4, tokenizer)
        acc_4 = (preds_4 == labels_4).mean()
        print("Label 3: ")
        preds_3, labels_3 = evaluate_llm(model, test_text_3, test_label_3, tokenizer)
        acc_3 = (preds_3 == labels_3).mean()
        print("Label 2: ")
        preds_2, labels_2 = evaluate_llm(model, test_text_2, test_label_2, tokenizer)
        acc_2 = (preds_2 == labels_2).mean()
        print("Label 1: ")
        preds_1, labels_1 = evaluate_llm(model, test_text_1, test_label_1, tokenizer)
        acc_1 = (preds_1 == labels_1).mean()
        print("Label 0: ")
        preds_0, labels_0 = evaluate_llm(model, test_text_0, test_label_0, tokenizer)
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

        for t, label in zip(test_instructions, test_label):
            if label == 1:
                pos_test_text.append(t)
                pos_test_label.append(label)
            else:
                neg_test_text.append(t)
                neg_test_label.append(label)

        print("Pos: ")
        pos_preds, pos_labels = evaluate_llm(model, pos_test_text, pos_test_label, tokenizer)
        pos_acc = (pos_preds == pos_labels).mean()
        print(pos_acc)
        print("Neg: ")
        neg_preds, neg_labels = evaluate_llm(model, neg_test_text, neg_test_label, tokenizer)
        neg_acc = (neg_preds == neg_labels).mean()
        print(neg_acc)
        print("Delta: ")
        print(pos_acc - neg_acc)
        print("Robust Acc: ")
        print((pos_acc + neg_acc) / 2)


def evaluate_llm(model, text, labels, tokenizer):
    results = []
    oom_examples = []
    used_labels = []

    for instruct, label in tqdm(zip(text, labels)):
        inputs = tokenizer(
            instruct, return_tensors="pt", truncation=True
        )
        input_ids = inputs.input_ids.cuda()
        attn_masks = inputs.attention_mask.cuda()

        with torch.inference_mode():
            try:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_masks,
                    max_new_tokens=20,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1e-3,
                    pad_token_id=tokenizer.eos_token_id,
                )
                result = tokenizer.batch_decode(
                    outputs.detach().cpu().numpy(), skip_special_tokens=True
                )[0]
                # print(result)
                result = int(result[len(instruct):][0])
            except:
                oom_examples.append(input_ids.shape[-1])
                continue

            results.append(result)
            used_labels.append(int(label))

    preds = np.array(results)
    labels = np.array(used_labels)

    return preds, labels


def main(args):
    # train_dataset, test_dataset = get_newsgroup_data_for_ft(
    #     mode="train", train_sample_fraction=args.train_sample_fraction
    # )

    train_dataset, test_dataset, no_concept_test_instructions, \
    no_concept_test_label, concept_test_instructions, concept_test_label \
        = get_sentiment_data_for_ft(method=args.method, dataset=args.dataset, concept=args.concept)

    print(f"Training samples:{train_dataset.shape}")
    print(f"Test samples:{test_dataset.shape}")

    print(train_dataset['instructions'][:2])
    print(train_dataset['labels'][: 2])

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_ckpt,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=args.dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    results_dir = f"experiments/classification-sampleFraction-{args.train_sample_fraction}_epochs-" \
                  f"{args.epochs}_rank-{args.lora_r}_dropout-{args.dropout}_dataset-{args.dataset}_concept-" \
                  f"{args.concept}_method-{args.method}"

    training_args = TrainingArguments(
        output_dir=results_dir,
        logging_dir=f"{results_dir}/logs",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=100,
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )

    max_seq_length = 512  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=training_args,
        dataset_text_field="instructions",
    )

    trainer_stats = trainer.train()
    train_loss = trainer_stats.training_loss
    print(f"Training loss:{train_loss}")

    peft_model_id = f"{results_dir}/assets"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    with open(f"{results_dir}/results.pkl", "wb") as handle:
        run_result = [
            args.epochs,
            args.lora_r,
            args.dropout,
            train_loss,
        ]
        pickle.dump(run_result, handle)
    print("Training Experiment over")

    print("Test on reviews wo concepts: ")

    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    llm_inference_pos_neg(no_concept_test_instructions, no_concept_test_label, model, tokenizer, args.dataset)
    print("Test on reviews with concepts: ")
    llm_inference_pos_neg(concept_test_instructions, concept_test_label, model, tokenizer, args.dataset)

    print("Test Experiment over")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--train_sample_fraction", default=0.99, type=float)
    parser.add_argument('--dataset', default="amazon-shoe-reviews", type=str)
    parser.add_argument('--concept', default="size", type=str)
    parser.add_argument('--method', default="original", type=str)

    args = parser.parse_args()
    main(args)
