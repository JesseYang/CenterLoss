#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing
import pdb
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils import summary
import tensorflow.contrib.slim as slim
import shutil
import inception_resnet_v1
import facenet
from reader_facenet import *
from cfgs.config import cfg


class Model(ModelDesc):

    def __init__(self, train_model):
        super(Model, self).__init__()
        self.train_model = train_model

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.image_size, cfg.image_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    def _build_graph(self, inputs):
        # with tf.device('/gpu:0'):
        image, label = inputs
      
        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=10)

        # image = (image - 127.5) / 128

        image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), image)

        prelogits, _ = inception_resnet_v1.inference(image, cfg.keep_probability, 
            phase_train=self.train_model, bottleneck_layer_size=cfg.feature_length, 
            weight_decay=cfg.weight_decay)

        logits = slim.fully_connected(prelogits, cfg.nrof_classes, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(cfg.weight_decay),
                scope='Logits', reuse=False)

        #feature for face recognition
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        feature = tf.identity(embeddings, name="FEATURE")
        
        # Add center loss
        if cfg.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label, cfg.center_loss_alfa, cfg.nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * cfg.center_loss_factor)

        # Add cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logits, name='cross_entropy_per_example')
        softmax_loss = tf.reduce_mean(cross_entropy, name='softmax_loss')
        # tf.add_to_collection('softmax_loss', softmax_loss)

        # Calculate the total losses
        center_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        # tf.summary.scalar('regularization_losses', regularization_losses)
        loss = tf.add_n([softmax_loss] + center_loss, name='loss')
 
        center_loss = tf.identity(center_loss, name='center_loss')
        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            add_moving_summary(loss, wd_cost)
            add_moving_summary(softmax_loss)
            # add_moving_summary(center_loss)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            add_moving_summary(softmax_loss)
            # add_moving_summary(center_loss)
            self.cost = tf.identity(loss, name='cost')
    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        # return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        return tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)

def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, cfg.shuffle)
    if isTrain:
        augmentors = [
            # imgaug.RandomCrop(crop_shape=448),
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            # imgaug.Clip(),
            # imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = []
    ds = AugmentImageComponent(ds, augmentors)
  
    # if isTrain:
    ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds

def get_config(args):
    dataset_train = get_data('train')
    # dataset_val = get_data('test')
    train_model = False
    if args.is_train == True:
        train_model = True
    callbacks=[
            ModelSaver(),
            # MinSaver('cost')
            # InferenceRunner(dataset_val, [
            #     ScalarStats('softmax_loss'),
            #     ScalarStats('center_loss'),
            #     # ClassificationError('loss')
            #     ]),
                # HyperParamSetterWithFunc('learning_rate',
                #                      lambda e, x: 1e-4 * (4.0 / 5) ** (e * 2.0 / 3) ),
           ScheduledHyperParamSetter('learning_rate',
                                      #orginal learning_rate
                                      [(0, 1e-1), (65, 1e-2), (77, 1e-3), (1000, 1e-4)]),
                                      #new learning_rate
                                     # [(0, 1e-2)]),
            HumanHyperParamSetter('learning_rate'),
        ]

    return TrainConfig(
        dataflow=dataset_train,
        model=Model(train_model),
        callbacks=callbacks,
        steps_per_epoch=1000,
        max_epoch=8000000,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--batch_size', default=90)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--is_train', help='need if train else not need', action='store_true')
    parser.add_argument('--log_dir', help='train_log name', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = int(args.batch_size) // NR_GPU
    
    if args.log_dir != None:
        if os.path.exists(os.path.join("train_log", args.log_dir)):
            shutil.rmtree(os.path.join("train_log", args.log_dir))
        logger.set_logger_dir(os.path.join("train_log", args.log_dir))
    else:
        logger.auto_set_dir()

    config = get_config(args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
