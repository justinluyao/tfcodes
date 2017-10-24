# http://zangbo.me/2017/07/05/TensorFlow_9/

import tensorflow as tf

def read_batch_cpm(tfr_path, img_size, hmap_size, input_channels, num_of_classes, batch_size=16, num_epochs=None):

    tfr_queue = tf.train.string_input_producer(tfr_path, num_epochs=num_epochs, shuffle=True)

#multi_threads raeading
    data_list = [read_and_decode_cpm(tfr_queue, img_size, input_channels, hmap_size, num_of_classes) for _ in range(2 * len(tfr_path))]

    batch_images, batch_labels, batch_images_orig = tf.train.shuffle_batch_join(data_list, batch_size=batch_size,
                                                                                           capacity=100 + 6 * batch_size,
                                                                                           min_after_dequeue=100,
                                                                                           enqueue_many=True,
                                                                                           name='batch_data_read')

    return batch_images, batch_centers, batch_labels, batch_images_orig


def read_and_decode_cpm(tfr_queue, img_size, hmap_size, input_channels, num_of_classes):
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    queue_images = []
    queue_center_maps = []
    queue_labels = []
    queue_orig_images = []

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'heatmaps': tf.FixedLenFeature([], tf.string)
                                       })

    # img_size = 128
    # center_radius = 11
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, input_channels])
    img = tf.cast(img, tf.float32)

    # img = img[..., ::-1]
    # img = tf.image.random_contrast(img, 0.7, 1)
    # img = tf.image.random_brightness(img, max_delta=0.9)
    # img = tf.image.random_hue(img, 0.05)
    # img = tf.image.random_saturation(img, 0.7, 1.1)
    # img = img[..., ::-1]

    # heatmap = tf.decode_raw(features['heatmaps'], tf.float32)
    heatmap = tf.reshape(features['heatmaps'], [hmap_size, hmap_size, num_of_classes])

    queue_images, queue_labels, queue_orig_images = preprocessing(img, heatmap)

    return queue_images, queue_labels, queue_orig_images
    # return preprocessed_img, preprocessed_center_maps, preprocessed_heatmaps, img

def pre_processing(img, heatmap):


    return img, heatmaps, img
