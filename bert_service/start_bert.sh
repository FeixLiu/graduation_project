#!/usr/bin/env bash
bert-serving-start -mask_cls_sep -device_map=1 -pooling_strategy=NONE -max_seq_len=64 -model_dir=/home/lya/graduation_project/data/bert_service/uncased_L-12_H-768_A-12/ -num_worker=4
