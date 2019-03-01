#!/usr/bin/env bash
bert-serving-start -mask_cls_sep -pooling_strategy=NONE -max_seq_len=128 -model_dir=/home/lya/graduation_project/data/bert_service/cased_L-24_H-1024_A-16/ -num_worker=2
