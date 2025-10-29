import pandas as pd
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
from feats_extraction import extract_all_conll_tags
from decode_gram import transform_with_label
from functools import partial
from spell_utils import decode_spelling_errors_in_a_sent
from read_preds_script import read_preds, read_preds_with_spaces

def apply_label(first, label, first_feats, first_lemma, first_pos):
    """
    implements operation (from `label`) to the initial token (`first`)
    """
    if label == "Keep":
        return first
    if label.startswith("Insert"):
        return f"{first} {label[len('Insert'):]}"
    if label.startswith("Delete"):
        return ""
    if label in ("Spell", "Split", "SpellAddHyphen"):
        return f"{first}<SPELL>"
    if label.startswith("Gram"):
        # uncomment for debug purposes
        # print(first, label, first_pos, first_lemma, first_feats, "\n",transform_with_label(word=first,
        #                             lemma=first_lemma,
        #                             labels=label[len("Gram$"):].replace("$", ","),
        #                             pos=first_pos,
        #                             feats=first_feats
        #                             ))
        # print()
        return transform_with_label(word=first,
                                    lemma=first_lemma,
                                    labels=label[len("Gram$"):].replace("$", ","),
                                    pos=first_pos,
                                    feats=first_feats
                                    )
    if label == "LowerCase":
        return first.lower()
    if label == "UpperCase":
        return first.capitalize()
    if label.startswith("ReplaceWith"):
        return label[len("ReplaceWith"):]
    if label == "HyphenToNull":
        return first.replace("-", "")
    if label == "NullToHyphen":
        return f"{first}<NULLTOHYPHEN>"
    if label == "Join":
        return f"{first}<JOIN>"
    if label == "AddDot":
        return first + "."
    print(f"ERROR: NO SUCH OPERATION {label}")
    return f"ERROR: NO SUCH OPERATION {label}"

def apply_to_row(refined_row, apply_label=apply_label):
    """
    input: tuple `refined_row` consists of 
                                - initial token `in_word`
                                - `labels` (may be several)
                                - token_feats
                                - token_lemma
                                - token_pos
    returns correction to the case based on labels
    """
    in_word, labels, token_feats, token_lemma, token_pos = refined_row
    labels = labels.split(">")
    for label in labels:
        in_word = apply_label(in_word, label, token_feats, token_lemma, token_pos)
    return in_word

def make_sent(tokens):
    """
    comprises sentence of tokens
    """
    tokens = [token.strip() for token in tokens if token != ""]
    sent = " ".join(tokens)
    sent = sent.replace("<JOIN> ", "")
    sent = sent.replace("<NULLTOHYPHEN> ", "-")
    return sent[len("<CLS> "):] if "<CLS>" in sent else sent

argument_parser = ArgumentParser()
argument_parser.add_argument("-p", "--preds_file", required=True)
argument_parser.add_argument("-f", "--feats_file", required=True)
argument_parser.add_argument("-o", "--out_file", default="OUTPUT")
argument_parser.add_argument("-J", "--json_config", required=True)
argument_parser.add_argument("-S", "--preds_with_spaces", action="store_true")


    
if __name__ == "__main__":
    tqdm.pandas() # для прогресс-бара при декодировании
    args = argument_parser.parse_args()
    # чтение каталога, модели, iam-токена
    with open(args.json_config, "r", encoding="utf8") as f:
        config = json.load(f)
    iam_token = config["iam_token"]
    folder_id = config["folder_id"]
    # чтение меток
    if args.preds_with_spaces:
        args.preds_file = read_preds_with_spaces(args.preds_file)
    data_dict = read_preds(args.preds_file)
    lemmas_info, pos_info, feats_info = extract_all_conll_tags(args.feats_file, 
                                                         rel_sents=data_dict["sentences"])  
    assert len(feats_info) == len(data_dict["tokens"]) == len(pos_info) == len(lemmas_info)
    pred_words = []
    print("Beginning decoding of non-spell tags..")
    for (tokens_list, labels_list, feats, lemmas, pos) in zip(data_dict["tokens"], data_dict["tags"], 
                                                              feats_info, lemmas_info, pos_info):
        prep_row = zip(tokens_list, labels_list, feats, lemmas, pos)
        pred_words.append(list(map(apply_to_row, tqdm(prep_row))))
    assert len(data_dict["tokens"]) == len(pred_words)
    data_dict["preds"] = pred_words
    df = pd.DataFrame(data_dict)[["tokens", "preds"]]
    print("Non-spell decoding complete!")
    # преобразуем списки токенов в предложения
    df["tokens"] = df["tokens"].apply(make_sent)
    df["preds"] = df["preds"].apply(make_sent)
    decode_spells = partial(decode_spelling_errors_in_a_sent, folder_id=folder_id, iam_token=iam_token)
    # декодирование опечаток
    print("Beginning decoding of spelling errors..")
    new_df = df.loc[df["preds"].str.contains("<SPELL>"), "preds"]
    df.loc[df["preds"].str.contains("<SPELL>"), "preds"] = df.loc[df["preds"].str.contains("<SPELL>"), 
                                                                  "preds"].progress_apply(decode_spells)
    print("Spelling decoding complete!")
    with open(args.out_file, 'w') as f:
        print(*df["preds"].tolist(), sep="\n", file=f)