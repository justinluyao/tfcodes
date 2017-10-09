# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:21:16 2017

@author: sunny
"""
import configs.config as cfg


# reading the tfrecords file
def tf_reader(tfrecords):
    filename_queue = tf.train.string_input_producer([tfrecords])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example, features={
        'input_data': tf.FixedLenFeature([], tf.string),
        'label_data': tf.FixedLenFeature([], tf.string)
    })
    input_data = features['input_data']
    label_data = features['label_data']
    input_data = tf.decode_raw(input_data, tf.float32)
    label_data = tf.decode_raw(label_data, tf.float32)

    input_data = tf.reshape(input_data, [cfg.FLAGS.input_height, cfg.FLAGS.input_width, cfg.FLAGS.input_dims])
    label_data = tf.reshape(label_data, [cfg.FLAGS.hmap_height, cfg.FLAGS.hmap_width, cfg.FLAGS.output_dim])
    print('reading data from TFrecords')

    return input_data, label_data
