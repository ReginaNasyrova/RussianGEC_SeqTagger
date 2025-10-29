### Предварительно нужно записать результаты морфосинтаксического анализа в файлы
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
    if relevant_sents is not None:
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

def extract_tags(parsed_sents):
    """
    Записывает тэги из файла .conllu в словари для каждого слова в каждом предложении
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
    tags_list=[[{"form": token["form"], "lemma": token["lemma"], "pos": token["upos"], "feats": token["feats"]} for token in sent[0]] for sent in parsed_sents]
    return tags_list

def extract_relevant_tags_from_original_sents(tags, indices):
    """
    Извлекаем метки нужных слов из исходных предложений
    Вход: tags: [[{"form": .., "lemma": ..,..}, {"form1": .., "lemma1": ..,..}]..[..]] 
                список из словарей тэгов для каждого слова в предложении для каждого предложения
          indices: список списков индексов ошибок для каждого предложения
    """
    relevant_tags = []
    for tag_list, ids in zip(tags, indices):
        if ids == [(-1,-1)]:
            relevant_tags.append(None)
            continue
        # curr_tags = []
        # for id_pair in ids:
        #     curr_tags.append([tag_list[num] for num in range(id_pair[0], id_pair[1])])
        curr_tags = [tag_list[start:end] for start, end in ids]
        relevant_tags.append(curr_tags)
    return relevant_tags

def extract_relevant_tags_from_correct_sents(tags, indices, corrections):
    """
    Извлекаем метки нужных слов из исправленных предложений
    Вход: tags: [[{"form": .., "lemma": ..,..}, {"form1": .., "lemma1": ..,..}]..[..]] 
                список из словарей тэгов для каждого слова в предложении для каждого предложения
          indices: список списков индексов ошибок для каждого предложения
          corrections: список списков исправллений для каждого предложения
    """
    relevant_tags = []
    for tag_list, ids, corrs in zip(tags, indices, corrections):
        if ids == [(-1,-1)]:
            relevant_tags.append(None)
            continue
        curr_tags = [] # тэги для предложения
        cumulative_gain = 0 ## для извлечения метки нужного слова нужно знать его индекс, а для этого --
                            ## насколько добавление предыдущих исправлений сдвинуло исходные индексы ошибок
        for x, id_pair in enumerate(ids):
            curr_tags.append([tag_list[num] for num in range(ids[x][0]+cumulative_gain, ids[x][0]+cumulative_gain+len(corrs[x].split()))])
            if id_pair[0] == id_pair[1] and corrs[x] != "": #вставка
                cumulative_gain += len(corrs[x].split())
            elif corrs[x] == "" and id_pair[1]>id_pair[0]: #удаление
                cumulative_gain -= (id_pair[1]-id_pair[0])
            elif corrs[x] != "" and id_pair[1]>id_pair[0]: #замена
                cumulative_gain += (len(corrs[x].split())-(id_pair[1]-id_pair[0]))

        relevant_tags.append(curr_tags)
    return relevant_tags

def extract_in_out_tags(data, file_name, corr_file_name, extract_tags_func=extract_tags,
                        extract_in_tags=extract_relevant_tags_from_original_sents,
                        extract_out_tags=extract_relevant_tags_from_correct_sents):
    """
    Возвращает метки ошибочных слов и слов-исправлений с помощью совмещения предыдущих функций
    """
    in_data = []
    with open(file_name, "r", encoding="utf8") as fin:
        for line in fin:
            in_data.append(line)
    in_tags = extract_tags_func(in_data)
    in_tags = extract_in_tags(in_tags, data["indices"])
    out_data = []
    with open(corr_file_name, "r", encoding="utf8") as fin:
        for line in fin:
            out_data.append(line)
    out_tags = extract_tags_func(out_data)
    out_tags = extract_out_tags(out_tags, data["indices"], data["corrections"])
    return in_tags, out_tags

def extract_all_conll_tags(filename, rel_sents=None, extract_tags=extract_conll_tags):
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

def extract_all_tags(filename, extract_tags=extract_tags):
    """
    Возвращает мрлфг метки всех слов в предложениях из файла name
    Нужно для инференса, когда неизвестно заранее, для каких именно слов
    могут потребоваться метки
    """
    in_data = []
    with open(filename, "r", encoding="utf8") as fin:
        for line in fin:
            in_data.append(line)
    return extract_tags(in_data)