import json

path = '../../data/marco/train_v2.1.json'
with open(path, 'r') as file:
    data = json.load(file)
for i in range(len(data['answers'])):
    i = str(i)
    print(data['answers'][i])
    print(data['query'][i])
    for j in range(len(data['passages'][i])):
        print(data['passages'][i][j]['passage_text'].encode('utf-8'))
    break
"""
total = len(data['answers'])
zero = 0
one = 0
two = 0
more = 0
para_total = 0
for i in range(total):
    i = str(i)
    temp = 0
    para_total += len(data['passages'][i])
    for j in range(len(data['passages'][i])):
        temp += int(data['passages'][i][j]['is_selected'])
    if temp == 0:
        zero += 1
    elif temp == 1:
        one += 1
    elif temp == 2:
        two += 1
    else:
        more += 1

print(para_total, float(para_total) / total)
print(zero, float(zero) / total)
print(one, float(one) / total)
print(two, float(two) / total)
print(more, float(more) / total)
"""
