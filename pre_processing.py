# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:21:16 2017

@author: sunny
"""

import cv2
# import file reader
import data_reader_write.tf_data_producer as tf_data

def generate_mini_batch(tfr_path, num_epochs, heatmap_size, num_of_classes,center_radius, batch_size):


    with tf.name_scope('Batch_Inputs'):
        tfr_queue = tf.train.string_input_producer(tfr_path, num_epochs=num_epochs, shuffle=True)

        # images, centers, labels, image_orig = read_and_decode_cpm(tfr_queue, img_size, num_joints, center_radius)

        data_list = [read_and_decode_cpm(tfr_queue, img_size, num_joints, center_radius) for _ in
                     range(2 * len(tfr_path))]

        batch_images, batch_centers, batch_labels, batch_images_orig = tf.train.shuffle_batch_join(data_list,
                                                                                                   batch_size=batch_size,
                                                                                                   capacity=100 + 6 * batch_size,
                                                                                                   min_after_dequeue=100,
                                                                                                   enqueue_many=True,
                                                                                                   name='batch_data_read')

        # batch_labels = tf.image.resize_bilinear(batch_labels, size=tf.constant((hmap_size,hmap_size), name='shape'))

    return batch_images, batch_centers, batch_labels, batch_images_orig


def read_and_decode():
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_center_maps = []
    queue_labels = []
    queue_orig_images = []

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image': tf.FixedLenFeature([], tf.string),
                                               'heatmaps': tf.FixedLenFeature(
                                                   [int(img_size * img_size * (num_joints + 1))], tf.float32)
                                           })

        # img_size = 128
        # center_radius = 11
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [img_size, img_size, 3])
        img = tf.cast(img, tf.float32)

        img = img[..., ::-1]
        img = tf.image.random_contrast(img, 0.7, 1)
        img = tf.image.random_brightness(img, max_delta=0.9)
        img = tf.image.random_hue(img, 0.05)
        img = tf.image.random_saturation(img, 0.7, 1.1)
        img = img[..., ::-1]

        # heatmap = tf.decode_raw(features['heatmaps'], tf.float32)
        heatmap = tf.reshape(features['heatmaps'], [img_size, img_size, (num_joints + 1)])

        # create centermap
        center_map = tf.constant((cpm_utils.make_gaussian(img_size, center_radius,
                                                          [int(img_size / 2), int(img_size / 2)])).reshape(
            (img_size, img_size, 1)), name='center_map')
        center_map = tf.cast(center_map, tf.float32)

        # merge img + centermap + heatmap
        merged_img_heatmap = tf.concat([img, center_map, heatmap], axis=2)

        # subtract mean before pad
        mean_volume = tf.concat((128 * tf.ones(shape=(img_size, img_size, 3)),
                                 tf.zeros(shape=(img_size, img_size, (num_joints + 1))),
                                 tf.ones(shape=(img_size, img_size, 1))), axis=2)

        merged_img_heatmap -= mean_volume

        # preprocessing
        preprocessed_merged_img_c_heatmap, _, _ = preprocess(merged_img_heatmap,
                                                             label=None,
                                                             crop_off_ratio=0.05,
                                                             rotation_angle=0.8,
                                                             has_bbox=False,
                                                             do_flip_lr=True,
                                                             do_flip_ud=False,
                                                             low_sat=None,
                                                             high_sat=None,
                                                             max_bright_delta=None,
                                                             max_hue_delta=None)

        padded_img_size = img_size  # * (1 + tf.random_uniform([], minval=0.0, maxval=0.3))
        padded_img_size = tf.cast(padded_img_size, tf.int32)

        # resize pad
        preprocessed_merged_img_c_heatmap = tf.image.resize_image_with_crop_or_pad(preprocessed_merged_img_c_heatmap,
                                                                                   padded_img_size, padded_img_size)
        preprocessed_merged_img_c_heatmap += tf.concat((128 * tf.ones(shape=(padded_img_size, padded_img_size, 3)),
                                                        tf.zeros(
                                                            shape=(padded_img_size, padded_img_size, (num_joints + 1))),
                                                        tf.ones(shape=(padded_img_size, padded_img_size, 1))), axis=2)
        preprocessed_merged_img_c_heatmap = tf.image.resize_images(preprocessed_merged_img_c_heatmap,
                                                                   size=[img_size, img_size])

        with tf.control_dependencies([preprocessed_merged_img_c_heatmap]):
            # preprocessed_img = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,0], [368,368,3])
            # preprocessed_center_maps = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,3], [368,368,1])
            # preprocessed_heatmaps = tf.slice(preprocessed_merged_img_c_heatmap, [0,0,4], [368,368,13])

            preprocessed_img, preprocessed_center_maps, preprocessed_heatmaps = tf.split(
                preprocessed_merged_img_c_heatmap, [3, 1, (num_joints + 1)], axis=2)

            # Normalize image value
            preprocessed_img /= 256
            preprocessed_img -= 0.5

            queue_images.append(preprocessed_img)
            queue_center_maps.append(preprocessed_center_maps)
            queue_labels.append(preprocessed_heatmaps)
            queue_orig_images.append(img)

    return queue_images, queue_center_maps, queue_labels, queue_orig_images
    # return preprocessed_img, preprocessed_center_maps, preprocessed_heatmaps, img

import configs.config as cfg


# reading the tfrecords file
def tf_reader(tfrecords):
    filename_queue = tf.train.string_input_producer([tfrecords],num_epochs=num_epochs, shuffle=True)
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
