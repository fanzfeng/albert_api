# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng
import numpy as np


class BertVector(object):
    def __init__(self):
        self.max_batch = 512
        self.bc = None
        self.connect()
        # 需要根据服务端引入的模型更改模型维度
        self.vec_dim = 312

    def connect(self):
        from bert_serving.client import BertClient
        self.bc = BertClient(ip='127.0.0.1', port=5555, port_out=5556)

    def predict_sentences(self, sentences):
        num = len(sentences)
        if num == 1:
            return self.bc.encode(sentences)
        elif num <= self.max_batch:
            return self.bc.encode(sentences)
        else:
            m, n = num // self.max_batch, num % self.max_batch
            if n > 0:
                m += 1
            res = [self.bc.encode(sentences[i * self.max_batch:min((i + 1) * self.max_batch, num)]) for i in range(m)]
            return np.concatenate(res, axis=0)

    def text2vec(self, sentences):
        try:
            return self.predict_sentences(sentences)
        except:
            self.connect()
            return self.predict_sentences(sentences)


if __name__ == "__main__":
    model = BertVector()
    print(model.text2vec(["我们去不去"]))
