from conllu import parse

def extract_conll_tags(parsed_sents, relevant_sents, to_add_cls=True):
    """
    Записывает леммы, части речи и морфологические метки из файла .conllu 
    """
    curr = ""
    lines = []
    for elem in parsed_sents:
        if elem != "\n":
            curr += elem
        else:
            lines.append(curr)
            curr = ""
    parsed_sents = [parse(sent) for sent in lines]
    # удалим из распаршенных предложения, исключенные при предобработке
    parsed_sents = [sent for sent in parsed_sents
                    if " ".join([token["form"] for token in sent[0]]) in relevant_sents]
    all_feats = []
    for sent in parsed_sents:
        sent_feats = []
        for token in sent[0]:
            if token["feats"]:
                token_feats = '|'.join([f"{key}={token['feats'][key]}" for key in token["feats"]])
            else:
                token_feats = None
            sent_feats.append(token_feats)
        all_feats.append(sent_feats)
    all_pos = [[token["upos"] for token in sent[0]] for sent in parsed_sents]
    all_lemmas = [[token["lemma"] for token in sent[0]] for sent in parsed_sents]
    if to_add_cls:
        all_feats = [["None"]+elem for elem in all_feats]
        all_pos = [["None"]+elem for elem in all_pos]
        all_lemmas = [["None"]+elem for elem in all_lemmas]
    # print(all_lemmas, all_feats)
    return all_lemmas, all_pos, all_feats

def extract_all_conll_tags(filename, rel_sents, extract_tags=extract_conll_tags):
    """
    Возвращает мрлфг метки всех слов в предложениях из файла name
    Нужно для инференса, когда неизвестно заранее, для каких именно слов
    могут потребоваться метки
    """
    in_data = []
    with open(filename, "r", encoding="utf8") as fin:
        for line in fin:
            in_data.append(line)
    return extract_conll_tags(in_data, relevant_sents=rel_sents)