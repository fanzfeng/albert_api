# -*- coding: utf-8 -*-
'''
基于官方接口estimator的Bert接口：
1）text classification
2) text2vector

简单说明：
  - 读取数据方法：①基于tf data批处理（bert官方源码的做法）   ②流式读取
  - 支持的model graph缓存（速度更快）
'''
import os, json
import time
import pickle

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from albert_zh import tokenization
from albert_zh import modeling
from albert_zh.featurizer import *

from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.framework import graph_util

learning_rate = 0.00005
gpu_memory_fraction = 0.8
max_seq_len = 32


class SimProcessor(DataProcessor):
    def __init__(self, label_file=None):
        if label_file and os.path.exists(label_file):
            with open(label_file, 'rb') as fp:
                self.id2label = {v:k for k, v in pickle.load(fp).items()}
        else:
            self.id2label = None    

    def get_sentence_examples(self, questions):
        if not isinstance(questions[0], (list, tuple)):
            questions = [[d, ] for d in questions]
        examples = []
        if len(questions[0]) > 1:
            for index, data in enumerate(questions):
                guid = 'test-%d' % index
                text_a = tokenization.convert_to_unicode(str(data[0]))
                text_b = tokenization.convert_to_unicode(str(data[1]))
                label = self.get_labels()[0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            for index, data in enumerate(questions):
                guid = 'test-%d' % index
                text_a = tokenization.convert_to_unicode(str(data[0]))
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=self.get_labels()[0]))
        return examples

    def get_labels(self):
        if self.id2label:
            return [self.id2label[k] for k in range(len(self.id2label))]
        return ['0', '1']


class BertApi(object):
    def __init__(self, api_server=True, model_type=["class", "vector"][0],
                 ckpt_dir="/Users/fanzfeng/Downloads/albert_TextAudit_e5_0518",
                 ckpt_file="model.ckpt-5650", conf_file='albert_config_tiny.json', vocab_file='vocab.txt'):
        self.mode = None
        self.tf_data_file = "data/data.tf"
        assert os.path.exists(ckpt_dir)
        self.max_seq_length = max_seq_len
        vocab_conf = vocab_file if os.path.exists(vocab_file) else os.path.join(ckpt_dir, vocab_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_conf, do_lower_case=True)
        conf = conf_file if os.path.exists(conf_file) else os.path.join(ckpt_dir, conf_file)
        self.bert_conf = modeling.BertConfig.from_json_file(conf)
        with open(conf, "r") as fp:
            self.vec_dim = json.load(fp)["hidden_size"]
        self.bert_path = ckpt_dir
        self.model_file = ckpt_file if os.path.exists(ckpt_file) else os.path.join(ckpt_dir, ckpt_file)
        self.processor = SimProcessor(os.path.join(ckpt_dir, "label2id.pkl"))
        tf.logging.set_verbosity(tf.logging.INFO)
        self.input_names = ['input_ids', 'input_mask', 'segment_ids']
        self.api_server = api_server
        self.model_type = model_type
        self.label_list = None if self.model_type == "vector" else self.processor.get_labels()
        self.id2label = self.processor.id2label if self.label_list else None
        self.graph_cache = True
        self.graph_file = os.path.join(ckpt_dir, "bert_model.pb")
        if self.graph_cache:
            self.optimize_graph()
        self.estimator = self.get_estimator()

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps,
                         num_warmup_steps, use_one_hot_embeddings):
        def model_fn(features, labels, mode, params):
            # tf.logging.info("*** Features ***")
            # for name in sorted(features.keys()):
            #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

            input_ids, input_mask, segment_ids = features["input_ids"], features["input_mask"], features["segment_ids"]
            model_output = self.task_model(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
            tvars = tf.trainable_variables()
            if init_checkpoint:
                (assignment_map, initialized_variable_names) \
                    = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if self.model_type != "class":
                tensor_pred = {"encode": model_output}
            else:
                tensor_pred = {"index": tf.argmax(model_output, axis=-1), 
                               "score": tf.reduce_max(model_output, axis=-1)}
            output_spec = EstimatorSpec(mode=mode, predictions=tensor_pred)
            return output_spec
        return model_fn

    def get_estimator(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.log_device_placement = False
        if self.graph_cache:
            def model_fn(features, labels, mode, params):
                with tf.gfile.GFile(self.graph_file, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                output = tf.import_graph_def(graph_def, input_map={k + ':0': features[k] for k in self.input_names},
                                             return_elements=['output:0'])
                if self.model_type != "class":
                    tensor_pred = {"encode": output[0]}
                else:
                    tensor_pred = {"index": tf.argmax(output[0], axis=-1), 
                                   "score": tf.reduce_max(output[0], axis=-1)}
                return EstimatorSpec(mode=mode, predictions=tensor_pred)
            return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

        label_list = self.processor.get_labels()
        model_fn = self.model_fn_builder(
            bert_config=self.bert_conf,
            num_labels=len(label_list),
            init_checkpoint=self.model_file,
            learning_rate=learning_rate,
            num_train_steps=None, num_warmup_steps=None, use_one_hot_embeddings=False)
        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config), model_dir=self.bert_path, 
                         params=dict())
    
    def optimize_graph(self):
        if os.path.exists(self.graph_file):
            return
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids, input_mask, segment_ids = [tf.placeholder(tf.int32, (None, self.max_seq_length), v) for v in 
                                                      self.input_names]
                model_output = self.task_model(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
                res = tf.identity(model_output, 'output')
                saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(self.bert_path))
                except:
                    saver.restore(sess, self.model_file)
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['output'])
        with tf.gfile.GFile(self.graph_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return

    def task_model(self, input_ids, input_mask, segment_ids):
        model = modeling.BertModel(
                config=self.bert_conf,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=False)
        if self.label_list is None:
            pooled = tf.squeeze(model.all_encoder_layers[-1][:, 0:1, :], axis=1)
        else:
            num_labels = len(self.label_list)
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
        
    def input_fn_builder(self, sentences, is_training=False, drop_remainder=False):
        processor = self.processor
        predict_examples = processor.get_sentence_examples(sentences)
        if not self.api_server:
            tf_file = self.tf_data_file
            file_based_convert_examples_to_features(predict_examples, processor.get_labels(), max_seq_len,
                                                    self.tokenizer, tf_file)
            name_to_features = {
                "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([], tf.int64),
                "is_real_example": tf.FixedLenFeature([], tf.int64), }

            def input_fn(params):
                # For training, we want a lot of parallel reading and shuffling.
                d = tf.data.TFRecordDataset(tf_file)
                if is_training:
                    d = d.repeat()
                    d = d.shuffle(buffer_size=100)
                # OOM when big model big batch[len(predict_examples)]
                d = d.apply(
                    tf.contrib.data.map_and_batch(
                        lambda record: tf.parse_single_example(record, name_to_features),
                        batch_size=5000,
                        drop_remainder=drop_remainder))
                return d
        else:
            def generate_from_input():
                features = convert_examples_to_features(predict_examples, processor.get_labels(), max_seq_len, 
                                                        self.tokenizer)
                yield {
                    'input_ids': [f.input_ids for f in features],
                    'input_mask': [f.input_mask for f in features],
                    'segment_ids': [f.segment_ids for f in features],
                    # 'label_ids': [f.label_id for f in features]
                }

            def input_fn():
                return (tf.data.Dataset.from_generator(
                    generate_from_input,
                    output_types={
                        'input_ids': tf.int32,
                        'input_mask': tf.int32,
                        'segment_ids': tf.int32,
                        # 'label_ids': tf.int32
                        },
                    output_shapes={
                        'input_ids': (None, self.max_seq_length),
                        'input_mask': (None, self.max_seq_length),
                        'segment_ids': (None, self.max_seq_length),
                        # 'label_ids': (1,)
                        }).prefetch(10))
        return input_fn

    def text2vec(self, sentences):
        feature_input = self.input_fn_builder(sentences)
        vec_list = [r["encode"] for r in self.estimator.predict(input_fn=feature_input, yield_single_examples=False)]
        return np.concatenate(vec_list, axis=0)

    def predict(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        # t1 = time.time()*1000
        if self.model_type == "class":
            feature_input = self.input_fn_builder(sentences)
            res = [r for r in self.estimator.predict(input_fn=feature_input, yield_single_examples=False)]
            if len(res) > 1:
                r_json = {"pred_socre": [self.id2label.get(ix, -1) for ix in np.concatenate([r["index"] for r in res],
                                                                                            axis=0)],
                          "pred_label": list(np.concatenate([r["score"] for r in res], axis=0)),
                          }
            else:
                r_json = {"pred_socre": [self.id2label.get(ix, -1) for ix in res[0]["index"]],
                          "pred_label": list(res[0]["score"]),}
        else:
            r_json = self.text2vec(sentences)
        # t2 = time.time()*1000
        # print("use time predict {}ms".format(t2-t1))
        return r_json


if __name__ == "__main__":
    from os.path import expanduser
    bert = BertApi(model_type=["class", "vector"][1],
                   ckpt_dir=os.path.join(expanduser("~"), "Downloads/albert_tiny_489k"),
                   ckpt_file="albert_model.ckpt",
                   conf_file='albert_config_tiny.json', vocab_file='vocab.txt')
    print(bert.vec_dim)
    print(bert.predict(["我们去"]))
