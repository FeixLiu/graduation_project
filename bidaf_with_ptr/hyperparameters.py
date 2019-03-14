class Hyperparameters():
    marco_train_path = '../../data/marco/train_v2.1.json'
    marco_eval_path = '../../data/marco/eval_v2.1_public.json'
    marco_dev_path = '../../data/marco/dev_v2.1.json'
    #word_5 = '../../data/word/my_word_dict_5'
    #word_50 = '../../data/word/my_word_dict_50'
    #word_100 = '../../data/word/my_word_dict_100'
    #word_200 = '../../data/word/my_word_dict_200'
    word = '../../data/word/my_word_dict_500'
    #word_1000 = '../../data/word/my_word_dict_1000'
    max_para = 10
    max_seq_length = 64
    bert_embedding_size = 768
    pos_weight = 0.1
    learning_rate = 0.0000001
    epoch = 10
    test_iter = 20