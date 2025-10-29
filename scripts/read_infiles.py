def read_synth_file(inp):
    """
    использовать с синтетическими данными
    """
    assert "generated" in inp
    sentences, corrections = [], []
    curr_corrs = []
    with open(inp, 'r', encoding='utf8') as fout:
        sentence = ''
        last = ""
        sents_count = 0
        for i, line in enumerate(fout):
            line = line.strip()
            if line == "":
                sents_count += 1
            if not (line.startswith('A ') and line[-2] == "|") and line != "": ## предложение
                if curr_corrs != []:
                    corrections.append('%'.join(curr_corrs))
                    curr_corrs = []
                sentences.append(line)
            if line.startswith('A ') and line[-2] == "|":
                curr_corrs.append(line)
            elif line == "" and not (last.startswith('A ') and last[-2] == "|"):
                curr_corrs.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
            last = line
        if last != "":
            sents_count += 1
        # print(sents_count)
        corrections.append('%'.join(curr_corrs))
    corrections = [corr.split('%') for corr in corrections]
    # print(sentences)
    assert len(corrections) == len(sentences)
    data_dict = {"sentences": [],"indices": [], "error_types": [], "corrections": [], "wrong_seqs": []}
    for i, (sentence, correction_list) in enumerate(zip(sentences, corrections)):
        data_dict["indices"].append([])
        data_dict["error_types"].append([])
        data_dict["corrections"].append([])
        data_dict["sentences"].append([sentence])
        data_dict["wrong_seqs"].append([])
        correction_list = sorted(correction_list, key = (lambda x: list(map(int, x[2:].split("|||")[0].split())) ))

        for correction in correction_list:
            # print(correction)
            indices, error_type, corr, _, _, _ = correction[2:].split("|||")
            a, b = map(int, indices.split())
            if a==b==-1:
                wrong_seq = ""
            else:
                wrong_seq = " ".join([sentence.split()[id] for id in range(a, b)])
            data_dict["indices"][-1] += [(a,b)]
            data_dict["error_types"][-1] += [error_type]
            data_dict["corrections"][-1] += [corr]
            data_dict["wrong_seqs"][-1] += [wrong_seq]


    return data_dict

def read_file(inp):
    """
    Возвращает словарь списков для файла {inp} с разметкой, как у gera
    (у безошибочных предложений есть аннотации)
    UPD: добавлена сортировка исправлений по индексам (ВАЖНО)
    """
    assert "RULEC" not in inp and "lang" not in inp
    sentences, corrections = [], []
    curr_corrs = []
    with open(inp, 'r', encoding='utf8') as fout:
        sentence = ''
        for line in fout:
            line = line.strip()
            if line.startswith('S '):
                if curr_corrs == [] and len(sentences) > 0:
                    curr_corrs.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
                if len(curr_corrs) > 0:
                    corrections.append('%'.join(curr_corrs))
                curr_corrs = []
                sentences.append(line[2:])
            if line.startswith('A '):
                curr_corrs.append(line)
        corrections.append('%'.join(curr_corrs))
    corrections = [corr.split('%') for corr in corrections]
    assert len(corrections) == len(sentences)
    data_dict = {"sentences": [],"indices": [], "error_types": [], "corrections": [], "wrong_seqs": []}
    for i, (sentence, correction_list) in enumerate(zip(sentences, corrections)):
        data_dict["indices"].append([])
        data_dict["error_types"].append([])
        data_dict["corrections"].append([])
        data_dict["sentences"].append([sentence])
        data_dict["wrong_seqs"].append([])
        # print(correction_list)
        correction_list = sorted(correction_list, key = (lambda x: list(map(int, x[2:].split("|||")[0].split())) ))

        for correction in correction_list:
            indices, error_type, corr, _, _, _ = correction[2:].split("|||")
            a, b = map(int, indices.split())
            if a==b==-1:
                wrong_seq = ""
            else:
                wrong_seq = " ".join([sentence.split()[id] for id in range(a, b)])
            data_dict["indices"][-1] += [(a,b)]
            data_dict["error_types"][-1] += [error_type]
            data_dict["corrections"][-1] += [corr]
            data_dict["wrong_seqs"][-1] += [wrong_seq]

    return data_dict

def add_dummy_corrections(data):
    answer = [{"sentences": [sent], "indices": [], "error_types": [], "corrections": [], "wrong_seqs": []} for sent in data]
    return {key: [elem[key] for elem in answer] for key in answer[0]}

def read_rulec_file(inp):
    """
    Возвращает словарь списков для файла {inp} с разметкой, как у rulec-gec
    (у безошибочных предложений нет аннотаций)
    """
    assert "RULEC" in inp or "lang" in inp
    sentences, corrections = [], []
    curr_corrs = []
    with open(inp, 'r', encoding='utf8') as fout:
        sentence = ''
        last = ""
        for i, line in enumerate(fout):
            line = line.strip()
            if line.startswith('S '):
                if curr_corrs != []:
                    corrections.append('%'.join(curr_corrs))
                    curr_corrs = []
                sentences.append(line[2:])
            if line.startswith('A '):
                curr_corrs.append(line)
            elif line == "" and last.startswith("S"):
                curr_corrs.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
            last = line
        corrections.append('%'.join(curr_corrs))
    corrections = [corr.split('%') for corr in corrections]
    assert len(corrections) == len(sentences)
    data_dict = {"sentences": [],"indices": [], "error_types": [], "corrections": [], "wrong_seqs": []}
    for i, (sentence, correction_list) in enumerate(zip(sentences, corrections)):
        data_dict["indices"].append([])
        data_dict["error_types"].append([])
        data_dict["corrections"].append([])
        data_dict["sentences"].append([sentence])
        data_dict["wrong_seqs"].append([])
        correction_list = sorted(correction_list, key = (lambda x: list(map(int, x[2:].split("|||")[0].split())) ))

        for correction in correction_list:
            indices, error_type, corr, _, _, _ = correction[2:].split("|||")
            a, b = map(int, indices.split())
            if a==b==-1:
                wrong_seq = ""
            else:
                wrong_seq = " ".join([sentence.split()[id] for id in range(a, b)])
            data_dict["indices"][-1] += [(a,b)]
            data_dict["error_types"][-1] += [error_type]
            data_dict["corrections"][-1] += [corr]
            data_dict["wrong_seqs"][-1] += [wrong_seq]


    return data_dict

def read_processed(filename):
    """
    Возвращает словарь со списками токенов, первых тэгов и вторых тэгов 
    для файла после предобработки 
    """
    data = {"tokens": [], "first_tags": [], "second_tags": []}
    with open(filename, "r", encoding="utf8") as inp:
        curr_tokens, curr_first_tags, curr_second_tags = [], [], []
        for line in inp:
            line = line.strip()
            if line == "":
                data["tokens"].append(curr_tokens)
                data["first_tags"].append(curr_first_tags)
                data["second_tags"].append(curr_second_tags)
                curr_tokens, curr_first_tags, curr_second_tags = [], [], []
                continue
            token, tags = line.split("\t")
            tags = tags.split(">")
            first_tag = [tags[0]]
            second_tags = tags[1:]
            curr_tokens.append(token)
            curr_first_tags.append(first_tag)
            curr_second_tags.append(second_tags)
        if curr_tokens and curr_first_tags and curr_second_tags:
            data["tokens"].append(curr_tokens)
            data["first_tags"].append(curr_first_tags)
            data["second_tags"].append(curr_second_tags)
    assert len(data["tokens"]) == len(data["first_tags"]) == len(data["second_tags"])
    return data