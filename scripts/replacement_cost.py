import pymorphy3
from rapidfuzz.distance import Levenshtein

morph_analyser = pymorphy3.MorphAnalyzer()

"""
 Функция calculate_replacement_cost()

 Вход: first_token: str, substitution: str,
        gamma: int (порог для расстояния Левенштейна (далее -- LD)),
        lemma_cost: float (награда за совпадение леммы, default=0.75),
        pos_cost: float (награда за совпадение части речи, default=0.5),
        morph_analyser,
        LD

 Выход: replacement_cost (далее -- RC): int

 RC = 1: first_token и substitution равны или отличаются только регистром и дефисами
 RC = -3: одно из слов -- знак пунктуации, а второе -- нет
 RC = 1/LD: LD(first_token, substitution) <= gamma
 RC = -1 (+lemma_cost) (+pos_cost): LD > gamma
 Совпадение и леммы, и части речи позволяет получить положительный скор
 (но меньший, чем если LD=2, например),
 совпадение же только части речи уменьшает штраф
"""

def calculate_replacement_cost(first_token, substitution, gamma,
                               lemma_cost=0.75, pos_cost=0.5,
                               morph_analyser=morph_analyser,
                               LD=Levenshtein.distance
                               ):
    puncts = '!?//.:;,–‒"...' # different dashes
    similarity = int(first_token == substitution)

    if similarity or \
        first_token.replace("-","").lower() == substitution.replace("-","").lower():
        replacement_cost = 1
    elif first_token in puncts and substitution not in puncts:
        replacement_cost = -3
    elif first_token not in puncts and substitution in puncts:
        replacement_cost = -3
    else:
        similarity = LD(first_token, substitution)
        if similarity <= gamma:
            replacement_cost = 1/similarity
        else:
            replacement_cost = -1
            parsed_first_token, parsed_substitution = \
                                        morph_analyser.parse(first_token)[0],\
                                        morph_analyser.parse(substitution)[0]

            first_token_lemma, first_token_pos = parsed_first_token.normal_form,\
                                                parsed_first_token.tag.POS
            substitution_lemma, substitution_pos = parsed_substitution.normal_form,\
                                                parsed_substitution.tag.POS
            if first_token_lemma == substitution_lemma:
                replacement_cost  += lemma_cost
            if first_token_pos == substitution_pos:
                replacement_cost += pos_cost
    return replacement_cost
