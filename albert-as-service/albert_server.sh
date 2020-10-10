#!/usr/bin/env bash
#@Author  : fanzfeng
bert-serving-start -model_dir /Users/fanzfeng/Downloads/albert_tiny_489k -ckpt_name albert_model.ckpt -config_name albert_config_tiny.json -cpu -graph_tmp_dir ../data
