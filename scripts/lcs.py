import re
from string import punctuation as PUNCTUATION
from collections import deque

import jsonlines

def decode(backtraces, m, n, d, return_index_alignment=False):
    answer, distances, alignment = [n], [], [(m, n)]
    while m > 0 or n > 0:
        (i, j), d = backtraces[(m, n)]
        if i < m:
            answer.append(j)
            distances.append(d)
        alignment.append((i, j))    
        m, n = i, j
    return answer[::-1], alignment[::-1], distances[::-1]

def is_simple_replacement(x, y):
    hyphens = ['-'] + [chr(x) for x in range(8210, 8213)]
    quotes = list('"«»') + ["``", "''"]
    return (
        (x.isspace() and y.isspace()) or (x in hyphens and y in hyphens) or
        x.lower().replace("ё", "е") == y.lower().replace("ё", "е") or
        (x in quotes and y in quotes)
    )

def is_space_position(first, i, second, j):
    # first[i] - пробел, а second[j] пробел или соседствует с нм
    if first[i].isspace():
        return j == len(second) or (i > 0 and first[i-1].isspace())
    return False

def find_alignment(first, second, discard_spaces=False, return_index_alignment=False):
    m, n = len(first), len(second)
    queue = deque([(0, 0, None, 0)])
    used = dict()
    while len(queue) > 0:
        i, j, prev, d = queue.popleft()
        if (i, j) in used:
            continue
        used[(i, j)] = (prev, d)
        not_best_path = False
        while i < m and j < n and (first[i] == second[j] or discard_spaces and is_simple_replacement(first[i], second[j])):
            i, j = i+1, j+1
            if (i, j) not in used:
                used[(i, j)] = (i-1, j-1), d
            else:
                not_best_path = True
                break
        if not_best_path:
            continue
        if i == m:
            if j == n:
                index_alignment, alignment, distances = decode(used, m, n, d)
                return (index_alignment if return_index_alignment else alignment, distances), d
            is_space = is_space_position(second, j, first, i)
            func = queue.appendleft if is_space and discard_spaces else queue.append
            func((i, j+1, (i, j), d+int(not is_space)))
        elif j == n:
            is_space = is_space_position(first, i, second, j)
            func = queue.appendleft if first[i].isspace() and discard_spaces else queue.append
            func((i+1, j, (i, j), d+int(not first[i].isspace())))
        else: # first[i] точно не равно second[j]
            queue.append((i+1, j+1, (i, j), d+1))
            if prev != (i, j-1):
                is_space = is_space_position(first, i, second, j)
                func = queue.appendleft if is_space and discard_spaces else queue.append
                func((i+1, j, (i, j), d+int(not is_space)))
            if prev != (i-1, j):
                is_space = is_space_position(second, j, first, i)
                func = queue.appendleft if is_space and discard_spaces else queue.append
                func((i, j+1, (i, j), d+int(not is_space)))

def make_word_ends(words):
    answer, last_end = {0: 0}, 0
    for i, word in enumerate(words, 1):
        last_end += len(word)+int(i < len(words))
        answer[last_end] = i
    return answer

def transform_alignment(indexes):
    answer = [(0, i) for i in range(indexes[0]+1)]
    for i, index in enumerate(indexes[1:], 1):
        if index > indexes[i-1]:
            answer.extend((i, k+1) for k in range(indexes[i-1], index))
        else:
            answer.append((i, indexes[i-1]))
    return answer

def extract_word_alignment_from_char(first, second):
    first_ends = make_word_ends(first)
    second_ends = make_word_ends(second)
    first_text, second_text = " ".join(first).lower(), " ".join(second).lower()
    (alignment, _), d = find_alignment(first_text, second_text, discard_spaces=True)
    # alignment = transform_alignment(index_alignment)
    answer = []
    for i, j in alignment:
        i_word, j_word = first_ends.get(i), second_ends.get(j)
        if i_word is not None and j_word is not None:
            answer.append((i_word, j_word))
    return answer

def extract_pivot_alignment(alignment):
    pivot_alignment = [0]
    while pivot_alignment[-1]+1 < len(alignment):
        pivot_pos = pivot_alignment[-1]
        while pivot_pos+1 < len(alignment):
            r, s = alignment[pivot_pos]
            if alignment[pivot_pos+1] == (r+1, s+1):
                break
            pivot_pos += 1
        if pivot_pos > pivot_alignment[-1]:
            pivot_alignment.append(pivot_pos)
        if pivot_pos+1 < len(alignment):
            pivot_alignment.append(pivot_pos+1)
    return [alignment[i] for i in pivot_alignment]
    

def find_word_alignment(first, second, is_equal_func=None):
    if is_equal_func is None:
        is_equal_func = (lambda x, y: False)
    m, n = len(first), len(second)
    queue = deque([(0, 0, None, 0)])
    used = dict()
    # ИЩЕМ НОП стандартным алгоритмом через очередь
    while len(queue) > 0:
        i, j, prev, d = queue.popleft()
        if (i, j) in used:
            continue
        used[(i, j)] = (prev, d)
        not_best_path = False
        while i < m and j < n and (first[i] == second[j] or is_equal_func(first[i], second[j])):
            i, j = i+1, j+1
            if (i, j) not in used:
                used[(i, j)] = (i-1, j-1), d
            else:
                not_best_path = True
                break
        if not_best_path:
            continue
        if i == m and j == n:
            index_alignment, alignment, _ = decode(used, m, n, d)
            break
        elif i == m:
            queue.append((i, j+1, (i, j), d+1))
        elif j == n:
            queue.append((i+1, j, (i, j), d+1))
        else: # first[i] точно не равно second[j]
            queue.append((i, j+1, (i, j), d+1))
            if prev != (i, j-1):
                queue.append((i+1, j, (i, j), d+1))
    # делаем посимвольное выравнивание тех участков текста, где есть более одного слова в источнике или в исправлении
    word_alignment = extract_pivot_alignment(alignment)
    answer = word_alignment[:1]
    for pos, (r, s) in enumerate(word_alignment[1:], 1):
        r_prev, s_prev = word_alignment[pos-1]
        if r-r_prev <= 1 and s-s_prev <= 1:
            answer.append((r, s))
        else:
            curr_alignment = extract_word_alignment_from_char(first[r_prev:r], second[s_prev:s])
            answer.extend([(r_prev+i, r_prev+j) for i, j in curr_alignment[1:]])
    return answer    



def find_corrections(first, second, discard_spaces=True, fout=None):
    (alignment, distances), cost = find_alignment(first.lower(), second.lower(), 
                                                  discard_spaces=discard_spaces, 
                                                  return_index_alignment=True)
    # print(len(first), len(alignment))
    # for start in range(0, len(alignment), 100):
    #     for r, (x, j) in enumerate(zip(first[start:start+100], alignment[start:start+100]), start):
    #         d = max(alignment[r+1]-j, 1)
    #         print(("_" if x.isspace() else x)+"*"*(d-1), end="", file=fout)
    #     print("", file=fout)
    #     for r, (x, j) in enumerate(zip(first[start:start+100], alignment[start:start+100]), start):
    #         # print(j, alignment[r+1])
    #         print(re.sub("\s", "_", second[j:alignment[r+1]]) if alignment[r+1] > j else "*", end="", file=fout)
    #     print("", file=fout)
    #     # print(",".join(map(str, distances[start:start+100])))
    positions = [0] + [alignment[i+1] for i, x in enumerate(first) if x == "\n"] + [len(second)]
    second_sents = [second[i:j] for i, j in zip(positions[:-1], positions[1:])]
    after_sent_distances = [0] + [distances[i+1] for i, x in enumerate(first) if x == "\n"] + [distances[-1]]
    sent_distances = [i-j for i, j in zip(after_sent_distances[1:], after_sent_distances[:-1])]
    return second_sents, alignment, sent_distances


if __name__ == "__main__":
    # source = "Собор Успения Пресвятой Богородицы входит в список ЮНЕСКО Всемирного Наследия \" Восьмое чудо света \" по мнению многих жителей Земли .\nЗаглавие : \" Собор поражающий своей красотой архитектуры \" ."
    # target = "Собор Успения Пресвятой Богородицы входит в список ЮНЕСКО «Всемирное наследие». По мнению многих жителей Земли, это «восьмое чудо света».\n\nЗаглавие: «Собор, поражающий своей красотой архитектуры»."
    # source_sents = re.split("\n+", source)
    # target_sents, alignment, sent_distances = find_corrections(source, target, discard_spaces=True)
    # for i, (first, second) in enumerate(zip(source_sents, target_sents)):
    #     print(first.strip())
    #     print(second.strip().replace("\n", "#"))
    #     print(sent_distances[i], f"{(sent_distances[i]/len(first.strip())):.3f}")
    data = list(jsonlines.open("dump/dump_0305_1556"))
    fout = open("dump/dump_0305_1556_processed", "w", encoding="utf8") # None
    for elem in data:
        source = elem["request"]["messages"][-1]["text"]
        target = elem['prediction']
        target_sents, _, sent_distances = find_corrections(source, target, discard_spaces=True, fout=fout)
        source_sents = re.split("\n+", source)
        # if len(re.split("\n+", target)) == len(source_sents):
        #     continue
        for i, (first, second) in enumerate(zip(source_sents, target_sents)):
            # print(i)
            print(first.strip(), file=fout)
            # print(i)
            print(second.strip().replace("\n", "#"), file=fout)
            print(sent_distances[i], f"{(sent_distances[i]/len(first.strip())):.3f}", file=fout)
        print("", file=fout)
        # print(f"{cost} {(cost / len(source)):.3f}", file=fout)
    if fout is not None:
        fout.close()
