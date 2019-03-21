class Hyperparameters():
    marco_train_path = '../../data/marco/train_v2.1.json'
    marco_eval_path = '../../data/marco/eval_v2.1_public.json'
    marco_dev_path = '../../data/marco/dev_v2.1.json'
    word = '../../data/word_embd/word_embd_500'
    max_para = 10
    max_seq_length = 64
    embedding_size = 300
    pos_weight = 0.1
    ptr_conv_beta = 0.5
    attention_inter_size = 256
    learning_rate = 0.0000001
    pos_para = 2
    epoch = 400
    vocab_size = 38579
    save_model_epoch = 100
    loss_acc_iter = 200
    max_to_keep = 10
    test_iter = 20
    keep_prob = 0.5