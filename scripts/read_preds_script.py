def read_preds(filename):
    """
    Файл с метками выглядит следующим образом (токен-предск.-истинные, без пробелов):
    <CLS>	Keep>Keep	Keep>Keep
    Сегодня	Keep>Keep	Keep>Keep
    ,	Keep>Keep	Keep>Keep
    наконец	Keep>Insert,	Keep>Keep
    набрав	Keep>Keep	Keep>Keep
    силы	Keep>Keep	Keep>Keep
    """
    data = {"tokens": [], "tags": []}
    with open(filename, "r", encoding="utf8") as inp:
        curr_tokens, curr_tags = [], []
        for line in inp:
            line = line.strip()
            if line == "":
                data["tokens"].append(curr_tokens)
                data["tags"].append(curr_tags)
                curr_tokens, curr_tags = [], []
                continue
            token, tags, _ = line.split("\t")
            curr_tokens.append(token)
            curr_tags.append(tags)
        if curr_tokens and curr_tags:
            data["tokens"].append(curr_tokens)
            data["tags"].append(curr_tags)
    assert len(data["tokens"]) == len(data["tags"])
    data["sentences"] = [" ".join(elem[1:]) for elem in data["tokens"]] # 1: -- avoiding cls
    return data

def read_preds_with_spaces(filename):
    with open(filename, "r", encoding="utf8") as f:
        curr = []
        lines = []
        for line in f:
            line = line.strip()
            if line == "":
                lines.append(curr)
                curr = []
                continue
            curr.append("\t".join(line.split("\t")[:3]))
    example = lines[4]
    join_count, hyphen_count = 0, 0
    pred_join_count, pred_hyphen_count = 0, 0
    all_data = []
    for example in lines:
        labs = [["<CLS>", "Keep>Keep", "Keep>Keep"]]
        for x, y in zip(example[1::2], example[2::2]):
            assert len(example[1::2]) == len(example[2::2])
            token, corr, pred = x.split("\t") # токены
            _, corr_, pred_ = y.split("\t") # пробелы 
            if corr == "Hyphenate": corr = "NullToHyphen"
            if pred == "Hyphenate": pred = "NullToHyphen"
            if corr == "RemoveHyphen": corr = "HyphenToNull"
            if pred == "RemoveHyphen": pred = "HyphenToNull"
            if corr == "AddHyphen": corr = "SpellAddHyphen"
            if pred == "AddHyphen": pred = "SpellAddHyphen"
            if corr.startswith("ChangeTo"): corr = f"ReplaceWith{corr[len('ChangeTo'):]}"
            if pred.startswith("ChangeTo"): pred = f"ReplaceWith{pred[len('ChangeTo'):]}"
            if corr_ in ["Join", "Hyphenate", "AddHyphen", "RemoveHyphen"]: corr_ = "Keep"
            if pred_ in ["Join", "Hyphenate", "AddHyphen", "RemoveHyphen"]: pred_ = "Keep"
            correct = f"{corr}>{corr_}"
            predicted = f"{pred}>{pred_}"
            # print(x, y)
            labs.append([token, predicted, correct])
        all_data.append(labs)
    total = []
    for string in all_data:
        labs = string
        for x, elem in enumerate(labs):
            # print(id, len(all_data))
            token, pred, true = elem
            # print(pred)
            pred, _ = pred.split(">")
            true, _ = true.split(">")
            if pred == "Join":
                pred_join_count += 1
            else:
                if pred_join_count > 1:
                    labs[x-1][1] = "Keep>"+">".join(labs[x-1][1].split(">")[1:])
                pred_join_count = 0    
            if true == "Join":
                join_count += 1
            else:
                if join_count != 0:
                    labs[x-1][2] = "Keep>"+">".join(labs[x-1][2].split(">")[1:])
                join_count = 0  
            if pred == "NullToHyphen":
                pred_hyphen_count += 1
            else:
                if pred_hyphen_count > 1:
                    labs[x-1][1] = "Keep>"+">".join(labs[x-1][1].split(">")[1:])
                pred_hyphen_count = 0    
            if true == "NullToHyphen":
                hyphen_count += 1
            else:
                if hyphen_count != 0:
                    labs[x-1][2] = "Keep>"+">".join(labs[x-1][2].split(">")[1:])
                hyphen_count = 0  
        labs = [f"{token}\t{pred}\t{true}" for token, pred, true in labs]  
        total.append(labs)
    with open("transformed_"+filename.split("/")[-1], "w", encoding="utf8") as fin:
        for line in total:
            print(*line, sep='\n', file=fin)
            print("", file=fin)
    return "transformed_"+filename.split("/")[-1]