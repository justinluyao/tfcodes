# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:21:16 2017

@author: sunny
"""

# Hyper parameters

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('TFrecords_dir', default_value=['/tfrecords/training.tfrecords'], docstring='Training data tfrecords')

tf.app.flags.DEFINE_string('pretrained_model',default_value='cpm_hand.pkl',docstring='Pretrained mode')

tf.app.flags.DEFINE_integer('input_height', default_value=256, docstring='Input image length')
tf.app.flags.DEFINE_integer('input_width', default_value=256, docstring='Input image width')
tf.app.flags.DEFINE_integer('input_dim', default_value=3, docstring='Input image channel')

tf.app.flags.DEFINE_integer('output_height', default_value=64, docstring='Output heatmap height')
tf.app.flags.DEFINE_integer('output_width', default_value=64, docstring='Output heatmap width')
tf.app.flags.DEFINE_integer('output_dim', default_value=64, docstring='Output heatmap dimension')

tf.app.flags.DEFINE_integer('stages', default_value=7, docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('center_radius', default_value=21, docstring='Center map gaussian variance')

tf.app.flags.DEFINE_integer('num_of_features', default_value=7, docstring='Number of joints')
tf.app.flags.DEFINE_integer('batch_size', default_value=1, docstring='Training mini-batch size')
tf.app.flags.DEFINE_integer('training_iterations', default_value=100000, docstring='Training iterations')

tf.app.flags.DEFINE_integer('lr', default_value=0.0005, docstring='Learning rate')
tf.app.flags.DEFINE_integer('lr_decay_rate', default_value=0.96, docstring='Learning rate decay rate')
tf.app.flags.DEFINE_integer('lr_decay_step',default_value=2000, docstring='Learning rate decay steps')

tf.app.flags.DEFINE_string('saver_dir', default_value='/saver/_cpm_hand_i' , docstring='Saved model name')
tf.app.flags.DEFINE_string('log_dir', default_value='/logs/_cpm_hand_i', docstring='Log directory name')
tf.app.flags.DEFINE_string('color_channel', default_value='RGB', docstring='dim')

tf.app.flags.DEFINE_integer('istraining', default_value=True, docstring='Learning rate decay steps')
tf.app.flags.DEFINE_integer('isbc', default_value=False, docstring='training on blue crystal')

