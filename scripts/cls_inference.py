import json
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from torch.utils.data import DataLoader
from functools import partial
from tqdm.auto import tqdm
from collate import collate_fn
from onemodel_seq2labels_dataset import Seq2Labels
from read_infiles import read_processed, read_rulec_file, read_file
from model_funcs import Detector
from onemodel_main import set_random_seed

def process_input(filename):
    lower_test_data = filename.lower() 
    if "preprocessed" in lower_test_data:
        data = read_processed(filename)
    elif "m2" in lower_test_data:
        if "lang" in lower_test_data or "rulec" in lower_test_data:
            data = {"tokens": [elem[0].split() for elem in read_rulec_file(filename)["sentences"]]}
        elif "gera" in lower_test_data:
            data = [elem[0] for elem in read_file(filename)["sentences"]]
    else: # if input_file is a list of sents
        data = []
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                data.append(line)
    return data

def predict_with_model(sample_loader, model, id2labels_dict, return_probs=False):
    def map_id_to_label(id, id2labels=id2labels_dict):
        return id2labels[id]
    answer, probs_answer = [], []
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(sample_loader)):
            output = model(**batch)
            for r, (labels, masks) in enumerate(zip(output["labels"], batch["labels_mask"])):
                labels = labels.cpu().numpy()[masks.cpu().numpy()]
                labels = list(map(map_id_to_label, labels))
                labels = ["Keep"] + labels #  for <CLS>
                answer.append(labels)
                if return_probs:
                    probs = torch.softmax(output["log_probs"][r], dim=-1).cpu().numpy()[masks.cpu().numpy()]
                    probs_answer.append(probs)
    answer = {"labels": answer} if not return_probs else {
                            "labels": answer, "probs": probs_answer
                                                        }
    return answer

argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--input_file", required=True)
argument_parser.add_argument("-o", "--out_file", default="preds")
# argument_parser.add_argument('-F', "--is_roberta", action="store_false", default=True)
# argument_parser.add_argument('-A', "--aggregation_type", default="first")
argument_parser.add_argument('-l', "--labels2id_file", required=True)
argument_parser.add_argument('-B', "--batch_size", default=16, type=int)
# argument_parser.add_argument('-L', "--lr", default=1e-5, type=float)
# argument_parser.add_argument('-T', "--add_token_type_embeddings", action="store_true")
# argument_parser.add_argument('-G', "--accumulation_step", default=1, type=int)
argument_parser.add_argument('-W', "--model_weights", required=True)
argument_parser.add_argument('-P', "--return_probs", action="store_true")
argument_parser.add_argument('-H', "--hyperparameters_config", required=True)


if __name__=="__main__":
    args = argument_parser.parse_args()
    set_random_seed(5)
    data = process_input(args.input_file)
    with open(args.hyperparameters_config, "r", encoding="utf8") as conff:
        hyperparams = json.load(conff)
    is_roberta = ("roberta" in hyperparams["encoder"].lower())
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["encoder"], 
                                                   add_prefix_space=True)
    dataset = Seq2Labels(tokenizer=tokenizer, data=data, is_roberta=is_roberta, 
                         aggregation_mode=hyperparams["aggregation_mode"])()
    valid_collator = partial(collate_fn, bert_tokenizer=tokenizer)
    loader = DataLoader(dataset.remove_columns(["tokens", "subtokens", "labels"]), 
                        batch_size=args.batch_size, 
                        shuffle=False, collate_fn=valid_collator)
    with open(args.labels2id_file, "r", encoding="utf8") as fin:
        LABELS2ID = json.load(fin)
    encoder = AutoModel.from_pretrained(hyperparams["encoder"]) if is_roberta else\
        T5EncoderModel.from_pretrained(hyperparams["encoder"])
    model = Detector(encoder=encoder, n_labels=len(LABELS2ID), lr=hyperparams["learning_rate"], 
                     is_roberta=is_roberta, 
                     add_token_type_embeddings=hyperparams["add_token_type_embeddings"],
                     accumulation_step=hyperparams["accumulation_step"], device="cuda", 
                     aggregation_mode=hyperparams["aggregation_mode"])
    model.load_state_dict(torch.load(args.model_weights))
    model_output = predict_with_model(sample_loader=loader, model=model,
                                      id2labels_dict={id: label 
                                                      for label, id in LABELS2ID.items()})
    preds = [['>'.join(elem[i:i+2]) for i in range(0, len(elem), 2)] for elem in model_output["labels"]]
    true_labels = [['>'.join((["Keep"]+elem)[i:i+2]) for i in range(0, len(elem), 2)] for elem in dataset["labels"]]
    # print(preds[0])
    if args.return_probs:
        probs = model_output["probs"]
    with open(args.out_file, "w", encoding="utf8") as outf:
        for tokens_list, preds_list, true_list in zip(dataset["tokens"], preds, true_labels):
            if tokens_list[0] != "<CLS>":
                tokens_list = ["<CLS>"] + tokens_list
            assert len(tokens_list) == len(preds_list) == len(true_list)
            for token, pred, true in zip(tokens_list, preds_list, true_list):
                print(f"{token}\t{pred}\t{true}", file=outf)
            print("", file=outf)