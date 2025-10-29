def apply_label(first, second, label):
    """
    implements operation (from `label`) to the initial token (`first`)
    returns true token (`second`) in case of spelling and grammatical errors
    """
    if label == "Keep":
        return first
    if label.startswith("Insert"):
        return f"{first} {label[len('Insert'):]}"
    if label.startswith("Delete") or label in ("M_Spell", "E_Spell"): ## !!!
        return ""
    if label in ("Spell", "Split", "SpellAddHyphen") or "Gram" in label:
        return second.split("|>")[0]
    if label == "B_Spell":
        return second.split("|>")[0]
    if label.startswith("B_ReplaceWith"):
        return label[len("B_ReplaceWith"):]
    if label.startswith("M_ReplaceWith") or label.startswith("E_ReplaceWith"):
        return ""
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
    return f"ERROR: NO SUCH OPERATION {label}"

def apply_to_row(refined_row):
        """
        input: tuple `refined_row` consists of 
                                    - initial token `in_word`
                                    - true token `out_word`
                                    - `labels` (may be several)
        returns correction to the case based on labels
        """
        in_word, out_word, labels = refined_row
        labels = labels.split(">")
        for label in labels:
            in_word = apply_label(in_word, out_word, label)
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

def get_right_tokens(tokens_list, inds_list, corrections):
    """
    comprises a correct sentence from 
                                    - sentence (`tokens_list`)
                                    - list of error indices from .m2 file (`inds_list`)
                                    - list of corrections from .m2 file (`corrections`)
    """
    tokens_with_inds = {(i, i+1): token for i, token in enumerate(tokens_list.split())}
    for corr, inds in zip(corrections, inds_list):
        if inds == (-1, -1):
            return tokens_list
        else:
            if inds[1]-inds[0] == 0:
                tokens_with_inds[(inds[0], inds[1])] = corr
            elif inds[1]-inds[0] == 1:
                tokens_with_inds[(inds[0], inds[1])] = corr
            elif inds[1]-inds[0] > 1:
                for i in range(inds[0], inds[1]):
                    del tokens_with_inds[(i, i+1)]
                tokens_with_inds[(inds[0], inds[1])] = corr
    tokens_with_inds = sorted(tokens_with_inds.items(), key = lambda x: x[0])
    tokens = [x[1] for x in tokens_with_inds if x[1] != ""]
    sent_final = " ".join(tokens).split() ## for erasing extra spaces
    sent_final = " ".join(sent_final)
    return sent_final
    
def verify_nonspell_nongram_labels(initial, final, labels, sentences, indices, correction_lists):
    """
    prints sentences which cannot be restored correctly with assigned labels
    
    `initial` -- list of lists of input tokens
    `final` -- list of lists of output tokens
    `labels` -- list of lists of labels
    `sentences` -- list of [sentences] from the read file
    `indices` -- list of lists with annotation indices for each sent from .m2 file
    `correction_lists` -- list of lists with corrections for each sent from .m2 file
    """
    pred_words = []
    for (a, b, c) in zip(initial, final, labels):
        prep_row = zip(a, b, c)
        pred_words.append(list(map(apply_to_row, prep_row)))
    pred_sents = list(map(make_sent, pred_words))
    true_sents = []
    for tokens, inds, corrs in zip(sentences, indices, correction_lists):
        true_sents.append(get_right_tokens(tokens[0], inds, corrs))
    for pred, true in zip(pred_sents, true_sents):
        if pred != true:
            print(f"pred\t{pred}")
            print(f"true\t{true}")
            print()
    print("Label Verification completed!")
    return