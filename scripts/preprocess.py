import configparser
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
from argparse import ArgumentParser
from read_infiles import read_file, read_rulec_file, read_synth_file
from morpho_tags import extract_all_tags
from alignment import align
from replacement_cost import calculate_replacement_cost
from verify_data_labels import verify_nonspell_nongram_labels
from functional_categories import is_preposition, is_conjunction
  
def add_alignment(data_dict, 
                  align=align):
    """
    `data_dict`: data dict from read file \n
    returns data dict with key "alignments" : tuple pairs of aligned erroneous tokens,\n
                                                [] for correct sents
    """
    corrections = data_dict["corrections"]
    wrong_seqs = data_dict["wrong_seqs"]
    alignments = []
    for corrs, wrongs in zip(corrections, wrong_seqs):
        if corrs == ["-NONE-"]:
            alignments.append([])
            continue
        assert len(corrs) == len(wrongs)
        curr_alignment = []
        for corr, wrong in zip(corrs, wrongs):
            curr_alignment.append(align(corr.split(), wrong.split()))
        alignments.append(curr_alignment)
    data_dict["alignments"] = alignments
    return data_dict

def refine_alignment(data_dict):
    """
    refines `data_dict`["alignments"] so that \n
    it contains alignment of all the tokens in a sent
    """
    sents, error_ids, alignments =\
        data_dict["sentences"], data_dict["indices"], data_dict["alignments"]
    for x, (tokens, indices, alignment) in enumerate(zip(sents, error_ids, alignments)):
        tokens = tokens[0].split() ## tokens is a [sent]
        if indices == [(-1,-1)]:
            data_dict["alignments"][x] = [("<CLS>", "<CLS>")] + [(token, token) for token in tokens]
            continue
        assert len(indices) == len(alignment) ## aligned pairs still for erroneous fragments only
        refined = []
        last = 0
        for pair_number, id_pair in enumerate(indices):
            a, b = id_pair
            correct_tokens = tokens[last:a]
            refined += [(token, token) for token in correct_tokens]
            refined += alignment[pair_number]        
            last = b
        refined += [(token, token) for token in tokens[last:]]
        data_dict["alignments"][x] = [("<CLS>", "<CLS>")] + refined
    return data_dict

def collapse_alignment(aligned):
    """
    drops aligned pairs with "<NULL>" as input token (insertion cases), \n
    uniting tokens for insertion with the previous non-null token
    """
    aligned = aligned[::-1]
    for i, elem in enumerate(aligned):
        if elem[0] == "<NULL>": ## elem[0] -- input token
            aligned[i+1] = (aligned[i+1][0], f"{aligned[i+1][1]}|>{elem[1]}")
    return [pair for pair in aligned[::-1] if pair[0] != "<NULL>"]

def map_in_feats(IN_info):
    """
    IN_info: tuple (list of input_tokens, list of morphological feats)
    applies after alignment collapse, 
    so there may be only ordinary tokens or tokens with spaces ("на счет") as input tokens
    maps morphological features to the first input token in a row
    """
    in_tokens, in_features = IN_info
    curr_ind = -1
    ins = []
    for in_token in in_tokens[1:]: ## in_tokens[0] = "CLS"
        curr_ind += 1
        try:
            ins.append(in_features[curr_ind])
        except:
            return "ERROR: SENTENCE CONTAINS OVERLAPPING ANNOTATIONS"
        if " " in in_token:
            curr_ind += len(in_token.split())-1
    assert curr_ind == len(in_features)-1
    ins = [None] + ins ## [None] for CLS token
    return ins

def map_out_feats(OUT_info):
    """
    `OUT_info`: tuple (list of output_tokens, list of morphological feats)
    output tokens may be ordinary tokens/NULL with/without spaces or insertions
    maps morphological features to the first output token in a row
    """
    out_tokens, out_features = OUT_info
    nonzero = -1
    outs = []
    for out_token in out_tokens:
        if not out_token.startswith("<NULL>") and not out_token.startswith("<CLS>"): ## может начаться с нуля, но дальше вставка
            nonzero += 1
            try:
                outs.append(out_features[nonzero])
            except:
                return "ERROR: SENTENCE CONTAINS OVERLAPPING ANNOTATIONS"
        else:
            outs.append(None)
        if "|>" in out_token:
            nonzero += len(out_token.split("|>"))-1
        if " " in out_token:
            for elem in out_token.split("|>"):  
                nonzero += len(elem.split())-1
    # print(out_tokens, outs, nonzero, len(out_features)-1, sep="\n")
    # print()
    assert nonzero == len(out_features)-1
    return outs

def get_replacement_operation(first, second, ins, outs):
    """
    returns type of replacement operation
    `ins`: dict of morphological features for `first`
    `outs`: dict of morphological features for `second`
    """
    if first == second:
        return "Keep"
    if first.lower() == second:
        return "LowerCase"
    if first.capitalize() == second:
        return "UpperCase"
    if second in '!?//.:;,–‒"...':
        return f"ReplaceWith{second}"
    if first.replace(" ", "-") == second: ## неверно раздельно, надо через дефис
        return "NullToHyphen"
    if first.replace("-", "") == second: #неверно через дефис, надо слитно
        return f"HyphenToNull"
    if second.replace("-", "") == first: #неверно слитно, надо через дефис
        return f"SpellAddHyphen"
    if second.replace(" ", "") == first: # неверно слитно, надо раздельно
        return f"Split"
    if first.replace(" ", "") == second: # неверно раздельно, надо слитно
        return "Join"
    if second == first + ".":
        return f"AddDot"
    if ins is not None and outs is not None:
        assert ((ins["form"] == first or first.startswith(ins["form"])) 
                        and (outs["form"] == second or second.startswith(outs["form"])))
        if ins["lemma"] == outs["lemma"]:
            different_tags = []
            if outs["feats"] is not None and ins["feats"] is not None:
                different_tags = [outs["feats"][key] for key in outs["feats"]
                                    if (key not in ins["feats"]
                                        or outs["feats"][key] != ins["feats"][key])]
            return "$".join(["Gram"]+different_tags) if different_tags else "Spell"
    if (is_preposition(first.lower()) and is_preposition(second.lower())) or\
        (is_conjunction(first.lower()) and is_conjunction(second.lower())):
        # print(f"preps/conj {first}, {second}")
        return f"ReplaceWith{second}"
    if calculate_replacement_cost(first, second, gamma=2) > 0:
        return "Spell"
    return f"ReplaceWith{second}"

def gather_label_in_a_row(row,
                          get_replacement_operation=get_replacement_operation,
                          is_conjunction=is_conjunction, is_preposition=is_preposition):
    """
    comprises final label for a `row`, 
                        which is a tuple of input_token(s), output_token(s),
                                            their morph. feats
    """
    in_word, out_word, in_feats, out_feats = row
    label = "Keep"
    out_words = []
    if "|>" in out_word:
        out_words = out_word.split("|>")
        out_word = out_words[0]
        out_words = out_words[1:]
    ## getting operation for the first output token
    if out_word == "<NULL>":
        label = "Delete"
        # if is_conjunction(in_word) or is_preposition(in_word):
        #     label = "DeleteFunct"
        # elif in_word in '!?//.:;,–‒"...':
        #     label = "DeletePunct"
        # else: label = "DeleteWord"
    else:
        label = get_replacement_operation(in_word, out_word,
                                          in_feats, out_feats)
    ## getting operation(s) for consequent insertion(s)
    if out_words:
        label = label + ">" + ">".join([f"Insert{token}" for token in out_words])
    return label

def process_alignment(data, add_alignment=add_alignment, refine_alignment=refine_alignment,
                      collapse_alignment=collapse_alignment):
    """
    returns aligned input and output tokens from data gathered from the read file
    """
    data = add_alignment(data)
    data = refine_alignment(data)
    data["alignments"] = list(map(collapse_alignment, data["alignments"]))
    input_tokens = [[elem[0] for elem in alignments] for alignments in data["alignments"]]
    output_tokens = [[elem[1] for elem in alignments] for alignments in data["alignments"]]
    return input_tokens, output_tokens

def process_sample(config_name, config_file, is_verify_labels,
                                    read_rulec_file=read_rulec_file,
                                    read_file=read_file, read_synth_file=read_synth_file,
                                    extract_all_tags=extract_all_tags,
                                    process_alignment=process_alignment,
                                    map_in_feats=map_in_feats, map_out_feats=map_out_feats,
                                    gather_label_in_a_row=gather_label_in_a_row,
                                    verify_nonspell_nongram_labels=verify_nonspell_nongram_labels):
    samplename = config_file["paths"]["samplename"]
    print(f"Starting processing of {samplename}..")
    if samplename == "RULEC-GEC" or samplename == "ru-lang8":
        sample_data = read_rulec_file(config_file["paths"]["data"])
    elif samplename == "GERA" or "rlc" in samplename:
        sample_data = read_file(config_file["paths"]["data"])
    elif "gen" in samplename or "3corps" in samplename:
        sample_data = read_synth_file(config_file["paths"]["data"])
    ## extraction of morphosyntactic tags
    sample_in_tags = extract_all_tags(filename=config_file["paths"]["parsed_data"])
    sample_out_tags = extract_all_tags(filename=config_file["paths"]["parsed_correct_data"])
    ## gathering aligned tokens
    sample_input_tokens, sample_output_tokens = process_alignment(sample_data)
    ## mapping morph. feats with tokens
    sample_data["in_feats"] = list(map(map_in_feats, zip(sample_input_tokens,
                                                            sample_in_tags)))
    sample_data["out_feats"] = list(map(map_out_feats, zip(sample_output_tokens,
                                                            sample_out_tags)))
    ## getting operation labels
    labels = []
    ambiguous_sents_ids = []
    for idx, (in_tokens, out_tokens, in_feats, out_feats) in enumerate(zip(
        sample_input_tokens, sample_output_tokens, sample_data["in_feats"], sample_data["out_feats"]
    )):
        if ("ERROR: SENTENCE CONTAINS OVERLAPPING ANNOTATIONS" in in_feats or
            "ERROR: SENTENCE CONTAINS OVERLAPPING ANNOTATIONS" in out_feats):
            ambiguous_sents_ids.append(idx)
            continue
        assert len(in_tokens) == len(out_tokens) == len(in_feats) == len(out_feats)
        labels.append(list(map(gather_label_in_a_row, zip(in_tokens, out_tokens, in_feats, out_feats))))
    
    labels = [[label+">Keep" if "Insert" not in label else label for label in labels_list] for labels_list in labels]
    ## getting rid of sents with ambiguous annotation
    ambiguous_sents = [sent[0] for i, sent in enumerate(sample_data["sentences"]) if i in ambiguous_sents_ids]
    sample_input_tokens = [elem for i, elem in enumerate(sample_input_tokens) if i not in ambiguous_sents_ids]
    sample_output_tokens = [elem for i, elem in enumerate(sample_output_tokens) if i not in ambiguous_sents_ids]  
    sample_data["sentences"] = [elem for i, elem in enumerate(sample_data["sentences"]) if i not in ambiguous_sents_ids]
    sample_data["indices"] = [elem for i, elem in enumerate(sample_data["indices"]) if i not in ambiguous_sents_ids]
    sample_data["corrections"] = [elem for i, elem in enumerate(sample_data["corrections"]) if i not in ambiguous_sents_ids]
    ## transforming input tokens with spaces ("так же") into separate tokens
    new_input_tokens, new_output_tokens, new_labels = [], [], []
    for sent_id, input_token_list in enumerate(sample_input_tokens):
        curr_inp_tokens, curr_out_tokens, curr_labels = [], [], []
        for token_id, input_token in enumerate(input_token_list):
            if " " not in input_token:
                curr_inp_tokens.append(input_token)
                curr_out_tokens.append(sample_output_tokens[sent_id][token_id])
                curr_labels.append(labels[sent_id][token_id])
                continue
            else:
                # print("ALIGNED")
                # print(input_token, sample_output_tokens[sent_id][token_id])
                first_token_label = labels[sent_id][token_id].split(">")[0] 
                space_labels = labels[sent_id][token_id].split(">")[1:]
                assert first_token_label in ("Join", "NullToHyphen", "Spell") or first_token_label.startswith("ReplaceWith") or first_token_label.startswith("Gram")# Spell для случаев На пример > Например,
                for word in input_token.split():
                    curr_inp_tokens.append(word)
                    curr_out_tokens.append(sample_output_tokens[sent_id][token_id])
                if first_token_label in ("Join", "NullToHyphen"):
                    curr_labels += [f"{first_token_label}>Keep"]*len(input_token.split()[:-1]) ## join/hyphentonull должно быть у всех токенов, кроме последнего
                    curr_labels.append(f"Keep>{'>'.join(space_labels)}")
                else: ## spell
                    curr_labels.append(f"B_{first_token_label}>Keep")
                    curr_labels += [f"M_{first_token_label}>Keep"]*len(input_token.split()[1:-1])
                    curr_labels.append(f"E_{first_token_label}>{'>'.join(space_labels)}")
        assert len(curr_inp_tokens) == len(curr_labels) == len(curr_out_tokens)
        new_input_tokens.append(curr_inp_tokens)
        new_output_tokens.append(curr_out_tokens)
        new_labels.append(curr_labels)
    sample_input_tokens, sample_output_tokens, labels =\
                        new_input_tokens, new_output_tokens, new_labels
    sample_data["labels"] = labels
    if is_verify_labels:
        verify_nonspell_nongram_labels(sample_input_tokens, sample_output_tokens, labels, sample_data["sentences"],
                                          sample_data["indices"], sample_data["corrections"])
    with open(f"{config_file['paths']['data_path']}preprocessed_{config_name.split('/')[-1]}.txt", 
              "w", encoding="utf8") as out_file:
        assert len(sample_input_tokens) == len(sample_data["labels"])
        for inps, labels, tokens in zip(sample_input_tokens, sample_data["labels"], sample_data["sentences"]):
            assert inps[1:] == tokens[0].split() ## проверка, что все токены исходного предложения на месте
            try:
                assert len(inps) == len(labels)
            except:
                print("ERROR: Different numbers of tokens and labels")
                print(inps)
                print(labels)
            for inp, label in zip(inps, labels):
                print(f"{inp}\t{label}", file=out_file)
            print("", file=out_file)
    print(f"{len(sample_data['sentences'])} sentences have been processed!")
    if ambiguous_sents:
        print("The following sentences have not been processed due to the ambiguity of annotation:")
        print(*[sent for sent in ambiguous_sents], sep="\n")
    print("Ending processing..")
    print(f"{len(ambiguous_sents)} ambiguous sentences..")
    print()
    return
    
argument_parser = ArgumentParser()
argument_parser.add_argument('-d', "--data_configs", nargs="+", required=True)
argument_parser.add_argument('-V', "--is_verify_labels", action="store_true")
argument_parser.add_argument('-o', "--output_file", default="OUTPUT.txt")
if __name__ == "__main__":
    args = argument_parser.parse_args()
    ## processing paths..
    for paths_config in args.data_configs:
        config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
        config.read(paths_config)
        process_sample(config_name=paths_config, config_file=config, is_verify_labels=args.is_verify_labels)