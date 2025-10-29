import re
from lcs import find_corrections
from yandex_chain import YandexLLM, YandexGPTModel

def make_request(text):
    prompt = "Дорогая модель, тебе будут даны слова с опечатками, в скобках будет указано предложение, в котором они встретились. Пожалуйста, выведи исправления этих слов в том же порядке, но без предложения в скобках и каких-либо комментариев, начиная со слова \"Ответ:\"."
    return f"{prompt}\n{text}"

def check_distance(result_word, source_word, max_dist=0.5, max_diff=2):
    if "—" in result_word.split():
        result_word = result_word.split(" — ")[-1]
    elif "—>" in result_word.split():
        result_word = result_word.split(" —> ")[-1]
    if "(" in result_word and ")" in result_word and not (len(result_word.split()) - len(source_word.split()) > 2): # 'ожидаемые (ожидаемой)'
        inner = re.search(r"(?<=\()[а-яА-ЯёЁ]+(?=\))", result_word)
        if inner:
            result_word = inner.group()
    if not "ya.ru" in result_word:
        predicted_sents, alignment, sent_distances = find_corrections(source_word, result_word, discard_spaces=True)
        predicted = predicted_sents[0]
        predicted = predicted.replace("*", "") # markdown
        relative_distance = sent_distances[0] / len(source_word)
        if (relative_distance > max_dist) or (len(result_word.split()) - len(source_word.split()) >= 2):
            predicted = source_word
    else: predicted = source_word
    return predicted

def decode_spelling_errors_in_a_sent(sentence, folder_id, iam_token, make_request=make_request):
    """
    Исправляет предложение с опечатками
    """
    # извлекаем токены, которым предсказана метка опечатки
    source_tokens = []
    for token in sentence.split():
        if token.endswith("<SPELL>"):
            source_tokens.append(token[:-len("<SPELL>")])
    # print(f"initial spell errors in {source_tokens}")
    query = f'{", ".join(source_tokens)} ({sentence.replace("<SPELL>", "")})'
    # print(f"query\t{query}")
    request = make_request(text=query)
    # print(f"request\t{request}")
    LLM = YandexLLM(model=YandexGPTModel.Pro, folder_id=folder_id, iam_token=iam_token, temperature=0.0)
    response = [token.strip().strip(".,") for token in LLM(request)[len("Ответ: "):].split(", ")]
    # print(f"LLM response\t{response}")
    if len(source_tokens) > len(response): # модель исправила меньше опечаток
        response += [response[-1]]*(len(source_tokens)-len(response))
    elif len(source_tokens) < len(response):
        response = response[:len(source_tokens)]
    assert len(source_tokens) == len(response)
    # fredxl предсказала одному токену delete>spell
    deleted_ids = [i for i, source_token in enumerate(source_tokens) if source_token == ""]
    source_tokens = [token for i, token in enumerate(source_tokens) if i not in deleted_ids]
    response = [token for i, token in enumerate(response) if i not in deleted_ids]
    # print(response)
    response = [check_distance(result_word=word, source_word=first_word)
                for word, first_word in zip(response, source_tokens)]
    # print(f"RES {response}")
    sent_with_spell = sentence.split()[::-1]
    for i, token in enumerate(sent_with_spell):
        if token.endswith("<SPELL>") and token != "<SPELL>":
            sent_with_spell[i] = response.pop().strip().replace("\n","") # pop, чтобы вышла ошибка, 
                                # если кол-во опечаток не совпадает с тем, что в ответе
        elif token.endswith("<SPELL>") and token == "<SPELL>":
            sent_with_spell[i] = ""
    return " ".join([elem for elem in sent_with_spell[::-1] if elem != ""])