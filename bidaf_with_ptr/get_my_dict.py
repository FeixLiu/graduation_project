from hyperparameters import Hyperparameters as hp
import json
import nltk
from tqdm import tqdm

word_count = {}

with open(hp.marco_dev_path) as file:
    data = json.load(file)

for i in tqdm(range(len(data['answers']))):
    query = data['query'][str(i)]
    token = nltk.word_tokenize(query)
    for j in token:
        try:
            word_count[j] += 1
        except KeyError:
            word_count[j] = 1
    answer = data['answers'][str(i)][0]
    token = nltk.word_tokenize(answer)
    for j in token:
        try:
            word_count[j] += 1
        except KeyError:
            word_count[j] = 1
    for m in range(len(data['passages'][str(i)])):
        text = data['passages'][str(i)][m]['passage_text']
        token = nltk.word_tokenize(text)
        for j in token:
            try:
                word_count[j] += 1
            except KeyError:
                word_count[j] = 1

with open(hp.marco_train_path) as file:
    data = json.load(file)

for i in tqdm(range(len(data['answers']))):
    query = data['query'][str(i)]
    token = nltk.word_tokenize(query)
    for j in token:
        try:
            word_count[j] += 1
        except KeyError:
            word_count[j] = 1
    answer = data['answers'][str(i)][0]
    token = nltk.word_tokenize(answer)
    for j in token:
        try:
            word_count[j] += 1
        except KeyError:
            word_count[j] = 1
    for m in range(len(data['passages'][str(i)])):
        text = data['passages'][str(i)][m]['passage_text']
        token = nltk.word_tokenize(text)
        for j in token:
            try:
                word_count[j] += 1
            except KeyError:
                word_count[j] = 1

word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

f5 = open(hp.word_5, 'w')
f50 = open(hp.word_50, 'w')
f100 = open(hp.word_100, 'w')
f200 = open(hp.word_200, 'w')
f500 = open(hp.word_500, 'w')
f1000 = open(hp.word_1000, 'w')
for i in word_count:
    if i[1] > 1000:
        try:
            print(i[0], file=f5)
            print(i[0], file=f50)
            print(i[0], file=f100)
            print(i[0], file=f200)
            print(i[0], file=f500)
            print(i[0], file=f1000)
        except UnicodeEncodeError:
            continue
    elif i[1] > 500:
        try:
            print(i[0], file=f5)
            print(i[0], file=f50)
            print(i[0], file=f100)
            print(i[0], file=f200)
            print(i[0], file=f500)
        except UnicodeEncodeError:
            continue
    elif i[1] > 200:
        try:
            print(i[0], file=f5)
            print(i[0], file=f50)
            print(i[0], file=f100)
            print(i[0], file=f200)
        except UnicodeEncodeError:
            continue
    elif i[1] > 100:
        try:
            print(i[0], file=f5)
            print(i[0], file=f50)
            print(i[0], file=f100)
        except UnicodeEncodeError:
            continue
    elif i[1] > 50:
        try:
            print(i[0], file=f5)
            print(i[0], file=f50)
        except UnicodeEncodeError:
            continue
    elif i[1] > 5:
        try:
            print(i[0], file=f5)
        except UnicodeEncodeError:
            continue

f5.close()
f50.close()
f100.close()
f200.close()
f500.close()
f1000.close()
