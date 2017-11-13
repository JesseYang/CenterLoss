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


from reader import *
from cfgs.config import cfg

class Model(ModelDesc):

    def __init__(self, depth):
        super(Model, self).__init__()
        self.depth = depth

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.img_h, cfg.img_w, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    def _build_graph(self, inputs):
        # with tf.device('/gpu:0'):
       
        image, label = inputs
       
        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=5)

        image = (image - 127.5) / 128

        image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        res_cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = res_cfg[self.depth]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            logits = (LinearWrap(image)
                    .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                    .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                    .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                    .apply(layer, 'group1', block_func, 128, defs[1], 2)
                    .apply(layer, 'group2', block_func, 256, defs[2], 2)
                    .apply(layer, 'group3', block_func, 512, defs[3], 2)
                    .BNReLU('bnlast')
                    .GlobalAvgPooling('gap')
                    .FullyConnected("fc1", out_dim=1024, nl=tf.identity)())

            s_net = (LinearWrap(logits)
                    .FullyConnected("fc2", out_dim=cfg.num_class, nl=tf.identity)())
        
        # logits = tf.sigmoid(logits) - 0.5
        feature = tf.identity(logits, name='FEATURE')

            
        # softmax-loss
        # net = tf.reshape(net, [-1])
        # feature = tf.identity(net, name="FEATURE")

        softmax_result = tf.nn.softmax(s_net)
        softmax_result = tf.identity(softmax_result, name="PRO")

        result_label = tf.reshape(s_net, (-1,cfg.num_class))
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_label, labels=label)
        # softmax_loss = tf.reduce_sum(softmax_loss)
        # pdb.set_trace()
        softmax_loss = tf.reduce_mean(softmax_loss, name="softmax_loss")
        # loss = tf.reduce_mean(softmax_loss, name="loss")

        #tensorboard
        # wrong = symbolic_functions.prediction_incorrect(result_label, label, name='inner')
        # train_error = tf.reduce_mean(wrong, name='train_error')
        # summary.add_moving_summary(train_error)

        # center-loss1
        centers = tf.get_variable('centers', [cfg.num_class, logits.get_shape()[1]], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1]) # from multi dim to 1 dim  
        centers_batch = tf.gather(centers, label)

        diff = cfg.center_loss_lr * (centers_batch - logits)
        centers = tf.scatter_sub(centers, label, diff)
        center_loss = tf.reduce_mean(tf.square(logits - centers_batch), name="center_loss")
        
        #center-loss2
        # center_loss, _ = center_loss_imp(net, label, cfg.center_loss_lr, cfg.num_class)


        # total loss
        loss = tf.add(softmax_loss, center_loss * cfg.center_loss_weight, name="loss")

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            add_moving_summary(loss, wd_cost)

            add_moving_summary(softmax_loss, center_loss)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            
            add_moving_summary(softmax_loss, center_loss)
            self.cost = tf.identity(loss, name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def get_data(train_or_test, square=False):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, is_square=square)
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
            imgaug.Clip(),
            imgaug.Flip(horiz=True),
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
    # pdb.set_trace()
    square = False
    if args.is_square:
    	square = True

    dataset_train = get_data('train', square)
    # dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
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
                                      #[(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
                                      #new learning_rate
                                     [(0, 1e-2)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(args.depth),
        # steps_per_epoch=1280000 / int(args.batch_size),
        max_epoch=10000000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--is_square', action='store_true', help='true for face square')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = int(args.batch_size) // NR_GPU
    
    logger.auto_set_dir()
    config = get_config(args)
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
