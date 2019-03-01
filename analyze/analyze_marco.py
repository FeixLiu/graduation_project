import json

path = '../../data/marco/train_v2.1.json'
with open(path, 'r') as file:
    data = json.load(file)
total = len(data['answers'])
max_text = 0
min_text = 9999
max_qas = 0
min_qas = 9999
max_ans = 0
min_ans = 9999
total_text = 0
text_num = 0
total_qas = 0
total_ans = 0
for i in range(total):
    i = str(i)
    query = data['query'][i].split(' ')
    answer = data['answers'][i][0].split(' ')
    total_ans += len(answer)
    total_qas += len(query)
    for j in range(len(data['passages'][i])):
        text = data['passages'][i][j]['passage_text'].split(' ')
        total_text += len(text)
        text_num += 1

print(text_num)

"""
print(total_text / text_num)    56.81811206271719
print(total_ans / total)        9.155245687379363
print(total_qas / total)        6.371066522737474
    
    if len(query) > max_qas:
        max_qas = len(query)
    if len(query) < min_qas:
        min_qas = len(query)
    if len(answer) > max_ans:
        max_ans = len(answer)
    if len(answer) < min_ans:
        min_ans = len(answer)
    for j in range(len(data['passages'][i])):
        text = data['passages'][i][j]['passage_text'].split(' ')
        if len(text) > max_text:
            max_text = len(text)
        if len(text) <  min_text:
            min_text = len(text)

print(max_text, min_text)   362 1
print(max_ans, min_ans)     322 1
print(max_qas, min_qas)     373 1

max_len = 0
min_len = 999
for i in range(len(data['answers'])):
    i = str(i)
    if len(data['passages'][i]) > max_len:
        max_len = len(data['passages'][i])
    if len(data['passages'][i])  < min_len:
        min_len = len(data['passages'][i])
print(max_len, min_len)
    print(data['answers'][i])
    print(data['query'][i])
    for j in range(len(data['passages'][i])):
        print(data['passages'][i][j]['passage_text'].encode('utf-8'))
    break
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
