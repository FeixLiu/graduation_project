path = '../../data/glove/glove.840B.300d.txt'
glove_embd = {}
with open(path, 'r', encoding='utf-8') as file:
    for line in file:
        row = line.strip().split(' ')
        glove_embd[row[0]] = row[1:]

path = '../../data/word/my_word_dict_500'
word_embd = {}
with open(path, 'r') as file:
    for line in file:
        line = line.strip()
        try:
            embd = glove_embd[line]
            word_embd[line] = embd
        except:
            continue

path = '../../data/word_embd/word_embd_500'
target = open(path, 'w')
for i in word_embd.keys():
    print(i, end=' ', file=target)
    for j in word_embd[i]:
        print(j, end=' ', file=target)
    print('\n', end='', file=target)





