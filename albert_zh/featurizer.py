# -*- coding: utf-8 -*-
import csv
import collections
import tensorflow as tf


class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_b) < 1:
      mid_ix = int(max_length/2)
      left_ix = max_length-mid_ix
      tokens_a = tokens_a[0:mid_ix]+tokens_a[-left_ix:]
    else:
      if len(tokens_a) > len(tokens_b):
        tokens_a.pop()
      else:
        tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
  if not label_list:
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  # assert len(input_ids) == max_seq_length
  # assert len(input_mask) == max_seq_length
  # assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label] if not label_list else None
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def single_text2feature(text, max_seq_length, tokenizer):
  tokens_a = tokenizer.tokenize(text)
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]
  tokens = []
  tokens.append("[CLS]")
  for token in tokens_a:
    tokens.append(token)
  tokens.append("[SEP]")
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
  return input_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index > 0 and ex_index % 10000 == 0:
        print("Writing example %d of %d" % (ex_index, len(examples)))
    feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
    features.append(feature)
  return features


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
  writer = tf.python_io.TFRecordWriter(output_file)
  for (ex_index, example) in enumerate(examples):
    if ex_index > 0 and ex_index % 10000 == 0:
      print("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,max_seq_length, tokenizer)
    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()
