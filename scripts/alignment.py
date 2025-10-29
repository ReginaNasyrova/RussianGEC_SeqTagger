import numpy as np
from replacement_cost import calculate_replacement_cost

"""
Два набора токенов: I, C
Матрица решений размера len(I)+1 * len(C)+1
операции: {'insert':1, 'delete':2, 'replace':3}
Матрица такого же размера для выбора оптимального выравнивания
Оптимальное выравнивание -- максимальный скор соответствия
i из I, c из C
первая строка и первый столбец матрицы скоров = -max(i,c)
0   -1  -2 ...
-1  ... ......
-2  ..........
вставка, удаление = -1
замена: см. calculate_replacement_cost()

 Функция align()

 Вход:  corrected_tokens: [token1:str, token2:str...],
        initial_tokens: [token1:str, token2:str...],
        gamma: int (порог для LD для calculate_replacement_cost, default=2),
        pad: str (что вставляется в выравнивание в случае вставки/удаления,
            default="<NULL>")

 Выход: [(initial_token1:str, correct_token1:str),..]
        с "0" на месте токенов в случае вставки/удаления
"""
def align(corrected_tokens, initial_tokens, gamma=2, pad="<NULL>", calculate_replacement_cost=calculate_replacement_cost):
    operations = [1,2,3]
    D_matrix = np.zeros((len(corrected_tokens)+1, len(initial_tokens)+1)).astype(int)
    S_matrix = D_matrix.copy()

    D_matrix[0,1:] = operations[1]
    D_matrix[1:,0] = operations[0]

    S_matrix[0,:] = range(len(initial_tokens)+1)
    S_matrix[0,:] *= -1
    S_matrix[:,0] = range(len(corrected_tokens)+1)
    S_matrix[:,0] *= -1

    insert_cost, delete_cost = -1, -1

    for c in range(1, len(corrected_tokens)+1):
        for i in range(1, len(initial_tokens)+1):
            in_token = initial_tokens[i-1]
            out_token = corrected_tokens[c-1]

            replace_cost = calculate_replacement_cost(in_token,
                                                      out_token,
                                                      gamma=gamma)

            options = np.array([S_matrix[c-1, i]+insert_cost,
                                S_matrix[c, i-1]+delete_cost,
                                S_matrix[c-1, i-1]+replace_cost])

            chosen_op, curr_score = np.argmax(options), np.max(options)
            S_matrix[c,i] = curr_score
            D_matrix[c,i] = operations[chosen_op]


    alignment = []
    i = len(initial_tokens)
    c = len(corrected_tokens)

    while  i!=0 or c!=0:
        if D_matrix[c,i] == 3:  # замена
            alignment.append((initial_tokens[i-1], corrected_tokens[c-1]))
            c -= 1
            i -= 1
        elif D_matrix[c,i] == 2: # удаление
            alignment.append((initial_tokens[i-1], pad))
            i -= 1
        elif D_matrix[c,i] == 1: # вставка
            alignment.append((pad, corrected_tokens[c-1]))
            c -= 1

    alignment = alignment[::-1]
    # print(alignment)
    """
    alignment: [(initial_token1: str, correct_token1: str),
                (initial_token2, correct_token2), ... ]
    """

    """
    Дополнительные шаги для объединения токенов в одну единицу выравнивания
    в случае неверно раздельного/слитного/дефисного написания.

    Категория 1:    initial: неверно написано раздельно
                    correct: должно писаться слитно/через дефис

    Категория 2:    initial: неверно написано слитно/через дефис
                    correct: должно писаться раздельно

    Категория 3:    initial: неверно написано слитно (через дефис)
                    correct: должно писаться через дефис (слитно)

    Случаи из категории 3 не рассматриваем, т.к. это соответствие вида
    один токен <-> один токен, и в нем алгоритм должен работать
    как с обычной заменой

    Ситуация несколько осложняется тем, что в выравнивании уже вставлены нули
    в случаев вставки/удаления, поэтому это придется учитывать

    """

    complex_token_indices = [] # индексы токенов, которые нужно объединить
                               # для категории 1
    new_complex_token_indices = [] # то же, но для случаев категории 2

    for i, (x, y) in enumerate(alignment):
        x, y = x.lower(), y.lower()
        if x != y:
            curr = i

            # [('0', 'за'), ('засчет', 'счет')]
            if x == pad.lower() and i != len(alignment)-1:
                x = alignment[i+1][0].lower()

            # [('в', '0'), ('добавок', 'вдобавок')]
            if y == pad.lower() and i != len(alignment)-1:
                y = alignment[i+1][1].lower()

            # Случаи категории 1, например, "тело сложению" вместо "телосложению"
            curr_list = []
            while y.startswith(x) and y != "" or (y[1:].startswith(x) and y[0]=="-"):
                # y[1:] to adress hyphenation

                y = y[len(x):]  # сдвигаем правильное слово на длину неправильного
                curr_list.append(curr) # записываем индекс пары
                if curr != len(alignment)-1:
                    x = alignment[curr+1][0] # если исходное слово не последнее,
                                            # берем следующее
                else:
                    break
                curr += 1

            if len(curr_list) > 1:
                complex_token_indices.append(curr_list)

            # то же самое, только для категории 2, например, "засчет" вместо
            #                                                   "за счет"
            a, b = y, x
            new_curr_list = []
            while b.startswith(a) and b != "" or (b[1:].startswith(a) and b[0]=="-"):
                b = b[len(a):]
                new_curr_list.append(curr)
                if curr != len(alignment)-1:
                    a = alignment[curr+1][1]
                else:
                    break
                curr += 1
            if len(new_curr_list) > 1:
                new_complex_token_indices.append(new_curr_list)
    """
     (new_) complex_token_indices: [[i1: int], [i2], ...]
     индексы (первое число) токенов, которые необходимо объединить
    """
    complex_token = ""

    # записываем в выравнивание случай из категории 1:
    # [..(f"{initial1} {initial2}}", f"{initial1}(-){initial2}")...]
    for elem in complex_token_indices:
        # берем кортежи из выравнивания с индексами из complex_token_indices
        # записываем в одну строку первые элементы кортежа -- исходные слова
        complex_token = " ".join([alignment[i][0] for i in elem])

        # записываем исправленное слово, которое должно быть слитным
        end_complex_token = alignment[elem[0]][1] \
                    if alignment[elem[0]][1] != pad else alignment[elem[0]+1][1]

        # удаляем из выравнивания позиции неверных частей и записываем новую пару
        alignment = [alignment[i] for i in range(len(alignment)) if i not in elem]
        alignment = alignment[:elem[0]] + [(complex_token, end_complex_token)] \
                                        + alignment[elem[0]:] # т.к. удалили элемент elem[0],
                                                            # то теперь это позиция
                                                            # следующего

    
    # аналогично работаем со случаями типа 2
    complex_token = ""
    for elem in new_complex_token_indices:
        complex_token = " ".join([alignment[i][1] for i in elem])
        start_complex_token = alignment[elem[0]][0] \
                    if alignment[elem[0]][0] != pad else alignment[elem[0]+1][0]
        alignment = [alignment[i] for i in range(len(alignment)) if i not in elem]
        alignment = alignment[:elem[0]] + [(start_complex_token, complex_token)]\
                                        + alignment[elem[0]:]
    return alignment
