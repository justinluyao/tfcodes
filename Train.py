# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:21:16 2017

@author: sunny
"""

######imports====================================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# import the config file according to the tasks
import configs.config as cfg

# import the model file according to the tasks
import depth_model

# import the pre_processing file
import pre_processing as prep

# training on the server
if cfg.FLAGS.isbc is False:
    import cv2
######imorts====================================================================================

if __name__ == '__main__':

    ######Pre-processing====================================================================================
    depth, hmap = tf_reader('training.tfrecords1')

    #    print('Reading Batch=====================================================================================')
    #


    depth_batch, hmap_batch = tf.train.shuffle_batch([depth, hmap],
                                                     batch_size=cfg.FLAGS.batch_size, capacity=10,
                                                     min_after_dequeue=5)

    ##
    ##    #
    ##    ##make input place holder========================================================================
    input_placeholder = tf.placeholder(dtype=tf.float32,
                                       shape=(cfg.FLAGS.batch_size, cfg.FLAGS.input_height, cfg.FLAGS.input_width, 3),
                                       name='input_placeholer')

    seg_placeholder = tf.placeholder(dtype=tf.float32,
                                     shape=(cfg.FLAGS.batch_size, cfg.FLAGS.hmap_height, cfg.FLAGS.hmap_width,
                                            cfg.FLAGS.num_of_classes), name='hmap_placeholder')
    ##    #
    ##    ##Building the model======================================================================
    model = depth_model.DEPTH_Model()
    model.build_model()
    model.build_loss()
    #    print('=====Model Build=====\n')
    ##    #
    ##    # # Training========================================================================
    with tf.Session() as sess:
        #    #
        coord = tf.train.Coordinator()
        #
        threads = tf.train.start_queue_runners(coord=coord)
        #
        #    #
        tf_w = tf.summary.FileWriter('./log', sess.graph)

        # Create model saver
        saver = tf.train.Saver(max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, '/saver/1200step-model.ckpt')

        while True:
            # Read in batch data
            batch_x_np, batch_y_np = sess.run([depth_batch, hmap_batch])
            #            print(np.sum(batch_x_np))
            #            print(np.argwhere(np.isnan(batch_y_np)))
            batch_y_np = np.nan_to_num(batch_y_np)

            stage_losses_np, total_loss_np, summary, current_lr, stage_heatmap_np, global_step = model.model_train(
                batch_x_np, batch_y_np, sess)
            #            print(np.shape(batch_y_np))
            #            print(np.shape(stage_heatmap_np[0]))
            #            ss = np.sum(stage_heatmap_np[-1],axis=-1)
            #            ss = np.sum(ss,axis=-2)
            #            ss = np.sum(ss)
            #            print(ss)
            #            print(np.max(stage_heatmap_np) )
            #            print(np.min(stage_heatmap_np) )

            # Write logs
            tf_w.add_summary(summary, global_step)
            # Draw intermediate results
            #             Draw intermediate results
            if global_step % 1 == 0:
                #
                demo_img = batch_x_np[0] + 0.5
                demo_stage_heatmaps = []
                for stage in range(cfg.FLAGS.stages):
                    print()
                    demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:cfg.FLAGS.num_of_joints].reshape(
                        (cfg.FLAGS.hmap_height, cfg.FLAGS.hmap_width, cfg.FLAGS.num_of_joints))
                    demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (cfg.FLAGS.input_height, cfg.FLAGS.input_width))
                    demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
                    demo_stage_heatmap = np.reshape(demo_stage_heatmap,
                                                    (cfg.FLAGS.input_height, cfg.FLAGS.input_width, 1))
                    #                   demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
                    demo_stage_heatmaps.append(demo_stage_heatmap)

                demo_gt_heatmap = batch_y_np[0, :, :, 0:cfg.FLAGS.num_of_joints].reshape(
                    cfg.FLAGS.hmap_height, cfg.FLAGS.hmap_width, cfg.FLAGS.num_of_joints)
                demo_gt_heatmap = cv2.resize(demo_gt_heatmap, (cfg.FLAGS.input_height, cfg.FLAGS.input_width))
                demo_gt_heatmap = np.amax(demo_gt_heatmap, axis=2)
                demo_gt_heatmap = np.reshape(demo_gt_heatmap, (cfg.FLAGS.input_height, cfg.FLAGS.input_width, 1))
                #                demo_gt_heatmap = np.repeat(demo_gt_heatmap, 3, axis=2)

                if cfg.FLAGS.stages > 4:
                    #                   upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]),
                    #                                              axis=1)
                    blend_img = 0.5 * demo_gt_heatmap + 0.5 * demo_img
                    demo_img = cv2.cvtColor(demo_img, cv2.COLOR_RGB2GRAY)

                    #                   blend_img = np.reshape(blend_img, (1,cfg.FLAGS.input_height, cfg.FLAGS.input_width, 1))

                    print(np.shape(blend_img))
                    #                   blend_img=blend_img[:,:,0]
                    lower_img = np.concatenate((demo_stage_heatmaps[cfg.FLAGS.stages - 1], demo_gt_heatmap),
                                               axis=1)
                    #                   demo_img = np.concatenate((blend_img*255, lower_img), axis=0)
                    cv2.imshow('demo', (demo_img * 255).astype(np.uint8))
                    cv2.waitKey(100)
                else:
                    upper_img = np.concatenate((cfg.FLAGS.stages - 1, demo_gt_heatmap, demo_img),
                                               axis=0)

                    #                 print(np.shape(upper_img))
                    #                 cv2.imshow('current heatmap', (demo_img * 255).astype(np.uint8))
                    #                 cv2.waitKey(100)

            ##
            print('##========Iter {:>6d}========##'.format(global_step))
            print('Current learning rate: {:.8f}'.format(current_lr))
            for stage_num in range(cfg.FLAGS.stages):
                print('Stage {} loss: {:>.3f}'.format(stage_num + 1, stage_losses_np[stage_num]))
            print('Total loss: {:>.3f}\n\n'.format(total_loss_np))

            #            # Save models
            model.save_model(global_step, saver, sess)
            #	         if global_step % 10 == 1:
            #	           with tf.device('/cpu:0'):
            ##                  save_path_str = os.path.join('D:/Projects/TFtraining/saver',str(global_step)+'.ckpt')
            #	             saver.save(sess=sess, save_path="/saver/model.ckpt",global_step=global_step)
            ##                  saver.save()
            #	             print('\nModel checkpoint saved...\n')

            # Finish training
            print(global_step)
            if global_step == 80000:
                break
                ##
        coord.request_stop()
        coord.join(threads)
        ##
        print('Training done.')
