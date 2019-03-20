import numpy as np
from load_dict import load_dict
from load_marco import load_marco
from bert import bert_server
from tqdm import tqdm
from hyperparameters import Hyperparameters as hp


def get_word_embd(words_embd):
    word_embd = []
    for i in range(len(words_embd)):
        word_embd.append(words_embd[i][1])
    while len(word_embd) < 64:
        word_embd.append([0. for _ in range(hp.bert_embedding_size)])
    return np.array(word_embd)


bert = bert_server()
vocab = load_dict(hp.word)
marco = load_marco(hp.marco_train_path, vocab, hp.max_seq_length, hp.max_para)

passage_embd = []
query_embd = []
answer_embd = []
answer_word_emnd = []
label = []
answer_index = []

for i in tqdm(range(marco.total)):
    passage_embd.append(bert.convert2vector(marco.passage[i]))
    query_embd.append(bert.convert2vector([marco.question[i] for _ in range(hp.max_para)]))
    answer_embd.append(bert.convert2vector([marco.answer[i]])[0])
    answer_word_emnd.append(get_word_embd(bert.convert2vector(marco.answer_word[i])))
    label.append(marco.label[i])
    answer_index.append(marco.answer_index[i])

passage_embd = np.array(passage_embd)
query_embd = np.array(query_embd)
answer_embd = np.array(answer_embd)
answer_word_emnd = np.array(answer_word_emnd)
label = np.array(label)
answer_index = np.array(answer_index)

np.save('../../data/marco_embd/marco_train_passage.npy', passage_embd)
np.save('../../data/marco_embd/marco_train_query.npy', query_embd)
np.save('../../data/marco_embd/marco_train_answer.npy', answer_embd)
np.save('../../data/marco_embd/marco_train_answer_word.npy', answer_word_emnd)
np.save('../../data/marco_embd/marco_train_label.npy', label)
np.save('../../data/marco_embd/marco_train_answer_index.npy', answer_index)
