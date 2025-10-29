import re

def transform_to_OC(tag, pos):
  """
  В PyMorphy используются метки OpenCorpora,
  а в анализаторе DeepPavlov -- UD Corpora.
  поэтому для использования PyMorphy нужно привести тэги к OpenCorpora
  """
  ##  Лицо
  if tag == '1':
    return '1per'
  if tag == '2':
    return '2per'
  if tag == '3':
    return '3per'

  ##  Формы глагола
  if tag == 'Ind':
    return 'indc'
  if tag == 'Imp':# and 'impr' in analyse.parse(corr_tags)[0].tag: ## Метки для императива и несов.в. одинаковые
    return 'impr'

  ##  Род
  if tag == 'Masc':
    return 'masc'
  if tag == 'Fem':
    return 'femn'
  if tag == 'Neut':
    return 'neut'

  ##  Залог
  if tag == 'Act':
    return 'actv'
  if tag == 'Pass':
    return 'pssv'

  ##  Вид -- (пока работать не будет)
  if tag == 'Imp':
    return 'impf'
  if tag == 'Perf':
    return 'perf'
  ##  Падеж
  if tag == 'Nom':
    return 'nomn'
  if tag == 'Gen':
    return 'gent'
  if tag == 'Dat':
    return 'datv'
  if tag == 'Acc':
    return 'accs'
  if tag == 'Ins':
    return 'ablt'
  if tag == 'Loc':
    return 'loct'
  if tag == 'Voc':
    return 'voct'
  if tag == 'Par':
    return 'gen2'

  ##  Время
  if tag == 'Pres':
    return 'pres'
  if tag == 'Past':
    return 'past'
  if tag == 'Fut':
    return 'futr'

  ##  Причастие
  if tag == 'Part':
    return 'PRTS' #PRTF -- full, PRTS -- short
  ##  Деепричастие
  if tag == 'Conv':
    return 'GRND'
  ##  Инфинитив
  if tag == 'Inf':
    return 'INFN'

  ##  Число
  if tag == 'Sing':
    return 'sing'
  if tag == 'Plur':
    return 'plur'

  ##  Формы прилагательного
  if tag == 'Short':
    if pos == 'VERB':
      return 'PRTS'
    else:
      return 'ADJS'
    
  if tag == 'Cmp':
    return 'COMP'
  if tag == 'Sup':
    return 'Supr'
  
  ## Одушевленность
  if tag in ['Anim', "Inan"]:
    return tag.lower()
  print(f"Warning: No such tag {tag}")
  return 'No such tag'

def read_file(filename, category):
    data = {"word": [], "lemma": [], "pos": [], "feats": [], "label": [], "answer": []}
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            word, lemma, tags, label, answer = line.split()
            # print(tags)
            tags = tags.split(",")
            pos_tag = tags[0]
            feats = None
            if tags[1:]:
              feats = tags[1]
            if category == "nonverb":
              if pos_tag == "VERB": ## сначала разберемся с остальным
                continue
            elif category == "verb":
              if pos_tag != "VERB":
                continue
            data["word"].append(word)
            data["lemma"].append(lemma)
            data["pos"].append(pos_tag)
            data["feats"].append(feats)
            data["label"].append(label)
            data["answer"].append(answer)
    return data
            
def calc_acc(true_words, pred_words, labels):
    assert len(true_words) == len(pred_words) == len(labels)
    all_words = len(pred_words)
    corr = 0
    for true, pred, label in zip(true_words, pred_words, labels):
        true, pred = true.lower(), pred.lower()
        true = re.sub('[ёЁ]', 'е', true)
        pred = re.sub('[ёЁ]', 'е', pred)
        if true.replace("ё", "е") == pred.lower().replace("ё", "е"):
            corr += 1
        else:
            print(f"true: |{true}|\tpred: |{pred}|\tlabel: {label}")
    print(f"accuracy\t{corr*100/all_words}")
    print(f"{corr} correct out of {all_words}")