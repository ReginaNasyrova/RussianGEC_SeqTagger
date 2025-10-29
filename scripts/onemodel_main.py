import json
import torch
import random
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from collections import Counter
from onemodel_seq2labels_dataset import Seq2Labels
from functools import partial
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from read_infiles import read_processed
from collate import collate_fn
from model_funcs import Detector, do_epoch

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_labels_dict(labels_lists, min_count=5):
    label_counts = {}
    for op_type_list in labels_lists:
        for elem in op_type_list:
            if elem not in label_counts:
                label_counts[elem] = 0
            label_counts[elem] += 1
    # label_counts = Counter([label for label_list in labels_lists for label in label_list ])
    labels = sorted([label for label, count in label_counts.items() if count >= min_count])
    label2ids = {label: i  for i, label in enumerate(labels)}
    id2labels = {i: label for label, i in label2ids.items()}
    return label2ids, id2labels

def map_sample_with_labels_ids(sample, labels2id_dict):
    """
    mapping labels in a sentence with their indices 
    """
    for elem in sample["labels"]:
        for x, tag in enumerate(elem):
            if tag in labels2id_dict:
                elem[x] = labels2id_dict[tag]
            else:
                elem[x] = labels2id_dict["Keep"]
    sample["labels"] = [torch.LongTensor(elem) for elem in sample["labels"]]
    return sample

def map_dataset(dataset, labels2id_dict, map_sample_with_labels_ids=map_sample_with_labels_ids):
    valid_mapper = partial(map_sample_with_labels_ids, labels2id_dict=labels2id_dict)
    dataset = dataset.map(valid_mapper, batched=True)
    return dataset
    
argument_parser = ArgumentParser()
argument_parser.add_argument('-t', "--train_samples", nargs="+", required=True)
argument_parser.add_argument('-v', "--val_samples", nargs="+", required=True)
argument_parser.add_argument('-o', "--output_dir", required=True)
argument_parser.add_argument('-l', "--labels2id_file", default=None)
argument_parser.add_argument('-F', "--is_roberta", action="store_false", default=True)
argument_parser.add_argument('-L', "--lr", default=1e-5, type=float)
argument_parser.add_argument('-B', "--batch_size", default=16, type=int)
argument_parser.add_argument('-A', "--aggregation_type", default="first")
argument_parser.add_argument('-M', "--mode", default="train")
argument_parser.add_argument('-T', "--add_token_type_embeddings", action="store_true")
argument_parser.add_argument('-G', "--accumulation_step", default=1, type=int)
argument_parser.add_argument('-W', "--weights", default=None)
argument_parser.add_argument('-E', "--nepochs", default=2, type=int)




if __name__=="__main__":
    set_random_seed(5)
    args = argument_parser.parse_args()
    encoder_name = "ai-forever/ruRoberta-large"
    if not args.is_roberta:
        encoder_name = "ai-forever/FRED-T5-1.7B" #"ai-forever/FRED-T5-large"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, 
                                                   add_prefix_space=True)
    if args.is_roberta:
        encoder = AutoModel.from_pretrained(encoder_name)
    else:
        encoder = T5EncoderModel.from_pretrained(encoder_name)
    ## processing paths..
    samples = []
    train_data = []
    for path in args.train_samples:
        sample = path.split('/')[-1].split("_")[-1][:-len(".txt")]
        print(f"Reading {sample}..")
        samples.append(sample)
        train_data.append(read_processed(path))
    train_datasets = concatenate_datasets([Seq2Labels(tokenizer=tokenizer, data=train_elem, 
                                                      is_roberta=args.is_roberta, 
                                                      aggregation_mode=args.aggregation_type
                                                      )() for train_elem in train_data])
    val_data = []
    for path in args.val_samples:
        val_sample = path.split('/')[-1].split("_")[-1][:-len(".txt")]
        print(f"Reading {val_sample}..")
        val_data.append(read_processed(path))
    val_datasets = concatenate_datasets([Seq2Labels(tokenizer=tokenizer, data=val_elem, 
                                                      is_roberta=args.is_roberta, 
                                                      aggregation_mode=args.aggregation_type
                                                      )() for val_elem in val_data])
    if args.mode == "train":
        LABELS2ID, _ = create_labels_dict(train_datasets["labels"])
        with open(f"{args.output_dir}{len(LABELS2ID)}_classes", "w", encoding="utf8") as ldict:
            json.dump(LABELS2ID, ldict)
    else:
        assert args.labels2id_file is not None
        with open(args.labels2id_file, "r", encoding="utf8") as fin:
            LABELS2ID = json.load(fin) 
    train_datasets = map_dataset(dataset=train_datasets, labels2id_dict=LABELS2ID)
    val_datasets = map_dataset(dataset=val_datasets, labels2id_dict=LABELS2ID)
    valid_collator = partial(collate_fn, bert_tokenizer=tokenizer)
    train_dataloader = DataLoader(train_datasets.remove_columns(["tokens", "subtokens"]), batch_size=args.batch_size, shuffle=True, 
                                  collate_fn=valid_collator)
    val_dataloader = DataLoader(val_datasets.remove_columns(["tokens", "subtokens"]), batch_size=args.batch_size, shuffle=False, 
                                  collate_fn=valid_collator)
    model = Detector(encoder=encoder, n_labels=len(LABELS2ID), lr=args.lr, is_roberta=args.is_roberta, 
                     add_token_type_embeddings=args.add_token_type_embeddings,
                     accumulation_step=args.accumulation_step, device="cuda", 
                     aggregation_mode=args.aggregation_type)
    if args.mode == "finetune":
        model.load_state_dict(torch.load(args.weights))
    best_val_sent_acc = 0.0
    checkpoint = "checkpoint.pt"
    sent_acc_dynamics, loss_dynamics, acc_dynamics = {i: 0.0 for i in range(args.nepochs)},\
        {i: 0.0 for i in range(args.nepochs)}, {i: 0.0 for i in range(args.nepochs)}
    for epoch in range(args.nepochs):
        if epoch == 0 and args.mode == "finetune":
            do_epoch(model, val_dataloader, mode="validate", epoch="evaluate") # валидация перед обучением, чтобы убедиться, 
                                                                                # что веса модели подгружены
        train_metrics = do_epoch(model, train_dataloader, mode="train", epoch=epoch+1)
        epoch_metrics = do_epoch(model, val_dataloader, mode="validate", epoch=epoch+1)
        sent_acc_dynamics[epoch], loss_dynamics[epoch], acc_dynamics[epoch] =\
            round(100*epoch_metrics["sent_accuracy"], 3), round(epoch_metrics["loss"], 3), round(100*epoch_metrics["accuracy"], 3)
        if epoch_metrics["sent_accuracy"] > best_val_sent_acc:
            best_val_sent_acc = epoch_metrics["sent_accuracy"]
            best_model_path = f"{args.output_dir}best_{checkpoint}"
            torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), f"{args.output_dir}epoch_{epoch+1}#_"+checkpoint) # > epoch+1

    model.load_state_dict(torch.load(best_model_path))
    ## финальная валидация лучшей модели
    final_metrics = do_epoch(model, val_dataloader, mode="validate", epoch="evaluate") # валидация
    with open(f"{args.output_dir}HyperParametersConfig.json", "w", encoding="utf8") as f:
        hyperparameters_dict = {
            "encoder": encoder_name,
            "n_labels": len(LABELS2ID),
            "aggregation_mode": model.aggregation_mode,
            "add_token_type_embeddings": model.add_token_type_embeddings,
            "train_batch_size": args.batch_size,
            "learning_rate": args.lr,
            "accumulation_step": model.accumulation_step,
            "mode": args.mode if args.mode == "train" else f"FT_from_{'/'.join(args.weights.split('/')[-2:])}",
            "val_loss": loss_dynamics,
            "val_acc": acc_dynamics,
            "val_sent_acc": sent_acc_dynamics,
            "train_data": args.train_samples,
            "val_data": args.val_samples
        }
        json.dump(hyperparameters_dict, f)