import pymorphy3
from gram_utils import read_file, transform_to_OC, calc_acc
from argparse import ArgumentParser

argument_parser = ArgumentParser()
argument_parser.add_argument("-d", "--data", default="all.morpho")
argument_parser.add_argument("-C", "--category", default="all")

def choose_right_analysis(in_token, token_lemma, dp_feats, analyse=pymorphy3.MorphAnalyzer()):
    ## Первые 23 варианта разбора 'её' - фиксированно 'её' во всех формах
    if in_token == 'её':
        return analyse.parse(in_token)[24]
    
    # если слова нет в словаре (вероятно, с опечаткой), то 
    # берется разбор для леммы, потому что она может быть определена корректно
    if str(analyse.parse(in_token)[0].methods_stack[0][0]) == 'FakeDictionary()':
        if in_token not in ["большинств"]:  # эта форма тоже считается ненастоящим, 
                                            # хотя по ней можно верно преобразовать слово
            in_token = token_lemma
    
    # определяем варианты разбора формы/леммы pymorphy
    analysis_variants = analyse.parse(in_token)
    
    # отдаем предпочтение разбору с совпадающей леммой
    analysis_variants = [analysis for analysis in analysis_variants if token_lemma == analysis.normal_form]
        
    # если нет разбора с той же леммой, довольствуемся имеющимся
    if not analysis_variants:
        analysis_variants = analyse.parse(in_token)
        
    # сортировка по количеству совпадающих граммем с разбором от deeppavlov
    analysis_variants = sorted(analysis_variants, key=lambda x:
                                            -len(set(x.tag.grammemes).intersection(set(dp_feats.values()))))
    
    # выбираем топ-разбор
    recommended = analysis_variants[0]
    
    # uncomment for debug purposes
    # for feat in dp_feats.values():
    #     if feat in ["inan", "anim"]:
    #         continue
        # if feat not in recommended.tag:
        #     print(in_token)
        #     print(token_lemma)
        #     print(dp_feats)
        #     print("REC", recommended)
        #     print(*analysis_variants, sep="\n")
        #     print()
        #     break 
    return recommended
    
def transform_with_label(word, lemma, labels, pos, feats, to_OC=transform_to_OC,
                         analyse=pymorphy3.MorphAnalyzer()):
    first_word_representation = word
    if feats is None or feats == "None":   # если метка предсказана предлогу или знаку препинания,
                        # то ничего не меняем
        return word
    
    # далее возможны вариации с исходной формой, поэтому сохраняем отдельно
    start_form = word
    
    # перевод признаков и меток в стандарт OpenCorpora
    feats = {tag_pair.split("=")[0]:to_OC(tag_pair.split("=")[1], pos=pos) 
                                for tag_pair in feats.split("|") if tag_pair not in ["Degree=Pos",
                                                                                     "VerbForm=Fin",
                                                                                     "Voice=Mid"]}
    labels = set([to_OC(label, pos=pos) for label in labels.split(",")])
    
    # в deeppavlov есть тэги, не имеющиеся в opencorpora, их удаляем
    to_del = []
    for label in labels:
        if label.startswith("No such tag"):
            to_del.append(label)
    for elem in to_del:
        labels.remove(elem)
    to_del = []
    for key, feat in feats.items():
        if feat.startswith("No such tag"):
            to_del.append(key)
    for elem in to_del:
        del feats[key]
    if not labels or not feats:
        return word
    
    if lemma == "счастие" and "ти" not in word:
        lemma = "счастье" 
        
    # часто исходный разбор не соответствует словоформе, а т.к. labels от него,
    # то оказываются к ней не применимы. Находим словоформу, которой соответствует
    # исходный разбор и преобразуем уже ее по метке вместо исходной словоформы.
    if analyse.parse(lemma)[0].inflect(set(feats.values())) and lemma != "который": 
        # исключаем который, потому что в признаках у deeppavlov нет числа
        if not (lemma == "один" and word.lower().startswith("перв")): # у форм слова "первый" лемма -- "один"
            word = analyse.parse(lemma)[0].inflect(set(feats.values())).word
    
    # выбор верного морфологического анализа
    analysis = choose_right_analysis(word, lemma, feats)
    
    # если при преобразовании в императив указывать лицо, то возвр-т None
    if 'impr' in labels:    
        if '1per' in labels:
            labels.remove('1per')
        elif '2per' in labels:
            labels.remove('2per')
            labels.add('excl') # для того чтобы, например, вместо "пойдемте" получилось "пойдите"
        elif '3per' in labels:
            labels.remove('3per')
        elif '2per' in list(feats.values()):
            labels.add('excl')
            
    # трансформация по метке
    answer = analysis.inflect(labels)

    # если не удалось изменить слово и остается только вернуть исходное
    if not answer or answer.word == start_form: 
        # ADJS/PRTS часто путаются
        if "ADJS" in labels and analysis.tag.POS == "PRTF":
            labels.add("PRTS")
            labels.remove("ADJS")
            answer = analysis.inflect(labels)
        elif "PRTS" in labels and analysis.tag.POS == "ADJF":
            labels.add("ADJS")
            labels.remove("PRTS")
            answer = analysis.inflect(labels)            
        # пробуем преобразовать по метке исходную словоформу
        answer = analyse.parse(start_form)[0].inflect(labels)
        
    if not answer: # по-прежнему ничего -- возвращаем исходное,
                   # иначе -- преобразованное
        return word
    else:
        answer = answer.word
        
    # сохранение того же регистра, что и у исходного слова
    if first_word_representation[0].isupper():
        answer = answer.title()
    # нормализация ё
    if "ё" in answer and not ("ё" in first_word_representation):
        answer = answer.replace("ё", "е")
    if "Ё" in answer and not ("Ё" in first_word_representation):
        answer = answer.replace("Ё", "Е")
    if answer == "годов":
        answer = "лет"
    if answer in ["мозге", "годе", "саде"]:
        answer = answer[:-1] + "у"
    return answer
    
    

if __name__ == "__main__":
    args = argument_parser.parse_args()
    data = read_file(args.data, category=args.category)
    transformed_words = [transform_with_label(
                            word=data["word"][i],
                            lemma=data["lemma"][i],
                            labels=data["label"][i],
                            pos=data["pos"][i],
                            feats=data["feats"][i])
                         for i in range(len(data["word"]))]
    calc_acc(true_words=data["answer"], pred_words=transformed_words, labels=data["label"])
    
    
    