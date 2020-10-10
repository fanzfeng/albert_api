import tensorflow as tf


class BertWorker():
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, graph_config):
        self.max_seq_len = args.max_seq_len
        self.do_lower_case = args.do_lower_case
        self.mask_cls_sep = args.mask_cls_sep
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.model_dir = args.model_dir
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.bert_config = graph_config
        self.use_fp16 = args.fp16
        self.show_tokens_to_client = args.show_tokens_to_client
        self.no_special_token = args.no_special_token

    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def, input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={'encodes': output[0]})

        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def run(self):
        estimator = self.get_estimator()
        for r in estimator.predict(self.input_fn_builder(receivers, tf, sink_token), yield_single_examples=False):
            # r['client_id'], r['encodes'].shapre
            return None

    def input_fn_builder(self, msg):
        from .bert.extract_features import convert_lst_to_features
        from .bert.tokenization import FullTokenizer

        def gen():
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'), do_lower_case=self.do_lower_case)

            is_tokenized = all(isinstance(el, list) for el in msg)
            tmp_f = list(convert_lst_to_features(msg, self.max_seq_len,
                                                    self.bert_config.max_position_embeddings,
                                                    tokenizer, logger,
                                                    is_tokenized, self.mask_cls_sep, self.no_special_token))
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              },
                output_shapes={
                    'input_ids': (None, None),
                    'input_mask': (None, None),
                    'input_type_ids': (None, None)}).prefetch(self.prefetch_size))

        return input_f


BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                 self.graph_path, self.bert_config)
