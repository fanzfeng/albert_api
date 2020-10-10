#!/usr/bin/env pyt
# -*- coding: utf-8 -*-
'''
基于静态图计算的bert接口
1) text classification
2) text2vector
'''

import os
import time
import json, pickle
import tensorflow as tf
# from keras.models import Model
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
from albert_zh import tokenization
from albert_zh import modeling
from albert_zh.featurizer import single_text2feature


class BertApi(object):
    def __init__(self, model_type=["class", "vector"][0],
                 ckpt_dir="/Users/fanzfeng/Downloads/albert_TextAudit_e5_0518",
                 ckpt_file="model.ckpt-5650",
                 conf_file='albert_config_tiny.json', vocab_file='vocab.txt', label_file="label2id.pkl"):
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.sess = tf.Session(config=config)
        self.max_seq_length = 32
        vocab_file = vocab_file if os.path.exists(vocab_file) else os.path.join(ckpt_dir, vocab_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        conf_file = conf_file if os.path.exists(conf_file) else os.path.join(ckpt_dir, conf_file)
        with open(conf_file, "r") as fp:
            self.vec_dim = json.load(fp)["hidden_size"]
        self.bert_conf = modeling.BertConfig.from_json_file(conf_file)
        self.bert_path = ckpt_dir
        self.model_file = ckpt_file if os.path.exists(ckpt_file) else os.path.join(ckpt_dir, ckpt_file)
        if model_type == "class":
            if not label_file:
               self.id2label = {0: "0", 1: "1"}
            else:
                label_file = label_file if os.path.exists(label_file) else os.path.join(ckpt_dir, label_file)
                if os.path.exists(label_file):
                    with open(label_file, 'rb') as fp:
                        self.id2label = {v: k for k, v in pickle.load(fp).items()}
                else:
                    raise ValueError("param label_file not exist")
        else:
            self.id2label = None
        self.input_ids, self.output = None, None
        self.model_type = model_type
        self.load_model()

    def task_model(self, input_ids):
        model = modeling.BertModel(
                config=self.bert_conf,
                is_training=False,
                input_ids=input_ids,
                input_mask=None,
                token_type_ids=None,
                use_one_hot_embeddings=False)
        if self.id2label is None:
            pooled = tf.squeeze(model.all_encoder_layers[-2][:, 0:1, :], axis=1)
        else:
            num_labels = len(self.id2label)
            output_layer = model.get_pooled_output()
            hidden_size = output_layer.shape[-1].value
            output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable("output_bias", [num_labels], 
                initializer=tf.zeros_initializer())
            with tf.variable_scope("loss"):
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                pooled = tf.nn.softmax(logits, axis=-1)
        return tf.cast(pooled, tf.float16)

    def load_model(self):
        with self.graph.as_default():
            #sess.run(tf.global_variables_initializer())
            self.input_ids = tf.placeholder(tf.int32, (None, self.max_seq_length), "input_ids")
            self.output = self.task_model(self.input_ids)
            # self.model = Model(input=self.input_ids, outputs=self.output)
            saver = tf.train.Saver()
            try:
                saver.restore(self.sess, tf.train.latest_checkpoint(self.model_file))
            except:
                saver.restore(self.sess, self.model_file)

    def predict(self, sentence):
        # t0 = time.time()
        with self.graph.as_default():
            feature = single_text2feature(sentence, self.max_seq_length, self.tokenizer)
            feed_dict = {self.input_ids: np.reshape([feature],(1, self.max_seq_length))}
            prob = self.sess.run([self.output], feed_dict)[0]
            # prob = self.model.predict(np.reshape([feature],(1, self.max_seq_length)))
        if self.model_type == "class":
            res = {"pred_label": [self.id2label.get(np.argmax(prob))],
                   "pred_score": [np.max(prob)]}
        else:    
            res = prob
        # t1 = time.time()
        # print("use time {}ms".format(1000*t1-1000*t0))
        return res


if __name__ == "__main__":
    from os.path import expanduser
    model = BertApi(model_type=["class", "vector"][1],
                    ckpt_dir=os.path.join(expanduser("~"),"Downloads/albert_tiny_489k"),
                    ckpt_file="albert_model.ckpt",
                    conf_file='albert_config_tiny.json', vocab_file='vocab.txt', label_file=None)
    print(model.predict("我们去不去"))
