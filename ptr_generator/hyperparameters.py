class Hyperparameters():
    marco_train_path = '../../data/marco/train_v2.1.json'
    marco_eval_path = '../../data/marco/eval_v2.1_public.json'
    marco_dev_path = '../../data/marco/dev_v2.1.json'
    glove_path = '../../data/glove/glove.840B.300d.txt'
    max_seq_length = 64
    attention_inter_size = 256
    prediction_inter_size = 1048576
    vocab_size = 2196010
    class_balance = -10
    batch_size = 4
    bert_embedding_size = 768
    keep_prob = 0.5
    learning_rate = 0.0001
    bidaf_lstm_hidden_units = 768
    test_iter = 200
    epoch = 100
    save_epoch = 20