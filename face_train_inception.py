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
    
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.img_size_h, cfg.img_size_w, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    def _build_graph(self, inputs):
        # with tf.device('/gpu:0'):
        image, label = inputs

        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=5)

        image = (image - 127.5) / 128
        
        # 5 * inception-resnet-A
        # def block35(resnet_a_net, scale=1.0):
        #     with argscope(Conv2D, kernel_shape=1, stride=1, nl=tf.identity, padding="SAME"):
        #         net1 = (LinearWrap(resnet_a_net)
        #             .Conv2D("conv35a", 32)())

        #         net2 = (LinearWrap(resnet_a_net)
        #             .Conv2D("conv35b1", 32)
        #             .Conv2D("conv35b2",  32, kernel_shape=3)())

        #         net3 = (LinearWrap(resnet_a_net)
        #             .Conv2D("conv35c1", 32)
        #             .Conv2D("conv35c2",  32, kernel_shape=3)
        #             .Conv2D("conv35c3", 32, kernel_shape=3)())
        #         mixed = tf.concat([net1, net2, net3], 3)

        #         up = (LinearWrap(mixed)
        #             .Conv2D("conv35c1", resnet_a_net.gat_shape()[3], nl=tf.nn.relu)())
        #         resnet_a_net += scale * up
        #     return resnet_a_net

        def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, \
                W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
            with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
                # pdb.set_trace()
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
                    tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
                mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
                up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')
                net += scale * up
                if activation_fn:
                    net = activation_fn(net)
            return net
        # 10 * inception-resnetp-B
        # def block17(resnet_b_net, scale=1.0):
        #     with argscope(Conv2D, kernel_shape=1, stride=1, nl=tf.identity, padding="SAME"):
        #         netb1 = (LinearWrap(resnet_b_net)
        #             .Conv2D("conv171", 128)())
        #         netb2 = (LinearWrap(resnet_b_net)
        #             .Conv2D("conv172", 128)
        #             .Conv2D("conv173", 128, kernel_shape=(1, 7))
        #             .Conv2D("conv174", 128, kernel_shape=(7, 1))())
        #         mixed = tf.concat([netb1, netb2], 3)
        #         up = (LinearWrap(mixed)
        #             .Conv2D("conv175", 896, nl=tf.nn.relu)())
        #         resnet_b_net += scale * up
        #     return resnet_b_net

        def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, \
                W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
            with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                                scope='Conv2d_0b_1x7')
                    tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                                scope='Conv2d_0c_7x1')
                mixed = tf.concat([tower_conv, tower_conv1_2], 3)
                up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')
                net += scale * up
                if activation_fn:
                    net = activation_fn(net)
            return net

        # def block8(resnet_c_net, scale=1.0, activation_fn = tf.nn.relu):
        #     with argscope(Conv2D, kernel_shape=1, stride=1, nl=tf.identity, padding="SAME"):
        #         netc1 = (LinearWrap(resnet_c_net)
        #             .Conv2D("conv81", 192)())
        #         netc2 = (LinearWrap(resnet_c_net)
        #             .Conv2D("conv82", 192)
        #             .Conv2D("conv83", 192, kernel_shape=(1,3))
        #             .Conv2D("conv84", 192, kernel_shape=(3,1))())
        #         mixed = tf.concat([netc1, netc2], 3)
        #         up = (LinearWrap(mixed)
        #             .Conv2D("conv85", net.get_shape()[3], nl=tf.nn.relu)())
        #         net += scale * up
        #         if activation_fn:
        #             net = activation_fn(net)
        #     return net
        def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, \
                W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
            with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
                with tf.variable_scope('Branch_1'):
                    tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                                scope='Conv2d_0b_1x3')
                    tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                                scope='Conv2d_0c_3x1')
                mixed = tf.concat([tower_conv, tower_conv1_2], 3)
                up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                                 activation_fn=None, scope='Conv2d_1x1')
                net += scale * up
                if activation_fn:
                    net = activation_fn(net)
            return net

        def block35repeat(l):
            # with tf.variable_scope("inception-resnet-a"):
            net = slim.repeat(l, 5, block35, scale=0.17)
            return net

        def block17repeat(l):
            # with tf.variable_scope("inception-resnet-b"):
            net = slim.repeat(l, 10, block17, scale=0.10)
            return net

        def block8repeat(l):
            # with tf.variable_scope("inception-resnet-c"):
            net = slim.repeat(l, 5, block8, scale=0.20)
            return net

        def reduction_a(l_out, k, l, m, n):
            with argscope(Conv2D, kernel_shape=3, nl=tf.identity, stride=2, padding="SAME", \
                W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
                reduce_a_1 = (LinearWrap(l_out)
                    .Conv2D("convreduction_a_1", n, padding="VALID")())

                reduce_a_2 = (LinearWrap(l_out)
                    .Conv2D("convreduction_a_2", k, kernel_shape=1, stride=1)
                    .Conv2D("convreduction_a_3", l, stride=1)
                    .Conv2D("convreduction_a_4", m, padding="VALID")())

                reduce_a_3 = (LinearWrap(l_out)
                    .MaxPooling("poool_a_1", shape=3, stride=2, padding="VALID")())

                net = tf.concat([reduce_a_1, reduce_a_2, reduce_a_3], 3)
            return net
            
        def reduction_b(b_out):
            with argscope(Conv2D, kernel_shape=3, nl=tf.identity, stride=1, padding="SAME", \
                W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
                reduce_b_1 = (LinearWrap(b_out)
                    .Conv2D("convreduction_b_1", 256, kernel_shape=1)
                    .Conv2D("convreduction_b_2", 384, stride=2, padding="VALID")())
                reduce_b_2 = (LinearWrap(b_out)
                    .Conv2D("convreduction_b_3", 256, kernel_shape=1)
                    .Conv2D("convreduction_b_4", 256, stride=2, padding="VALID")())
                reduce_b_3 = (LinearWrap(b_out)
                    .Conv2D("convreduction_b_5", 256, kernel_shape=1)
                    .Conv2D("convreduction_b_6", 256)
                    .Conv2D("convreduction_b_7", 256, stride=2, padding="VALID")())
                reduce_b_4 = (LinearWrap(b_out)
                    .MaxPooling("pooling_b1", shape=3, stride=2, padding="VALID")())

                # ConcatWith(x,tensor,dim)
                net = tf.concat([reduce_b_1, reduce_b_2, reduce_b_3, reduce_b_4], 3)
            return net

        def avg_pooling(l):

            net = slim.avg_pool2d(l, l.get_shape()[1:3], padding='VALID',scope='AvgPool_1a_8x8')
            net = slim.flatten(net)
            net = slim.dropout(net, 0.8, is_training=True, scope='Dropout')
            net = slim.fully_connected(net, 128, activation_fn=None, scope='Bottleneck', reuse=False)
            return net
        def center_loss_imp(features, labels, alpha, num_class):
            num_features = features.get_shape()[1]
            centers = tf.get_variable('centers', [num_class, num_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
            
            labels = tf.reshape(labels, [-1])
            centers_batch = tf.gather(centers, labels)

            center_loss = tf.nn.l2_loss(features - centers_batch, name="center_loss")##center_loss
            # center_loss = tf.div(tf.nn.l2_loss(features - centers_batch), int(num_features), name="center_loss")##center_loss
            #update centers
            diff = centers_batch - features
            unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
            appear_times = tf.gather(unique_count, unique_idx)
            appear_times = tf.reshape(appear_times, [-1, 1])

            diff = diff / tf.cast((appear_times + 1), tf.float32)
            diff = alpha * diff
            centers = tf.scatter_sub(centers, labels, diff)

            return center_loss, centers


        with argscope(Conv2D, kernel_shape=3, nl = tf.nn.relu, stride=1, padding="SAME", \
            W_init = tf.truncated_normal_initializer(stddev=0.1), use_bias=False):
            net = (LinearWrap(image)
                .Conv2D("convs1", 32, stride=2, padding="VALID")
                .Conv2D("convs2", 32, padding="VALID")
                .Conv2D("convs3", 64)
                .MaxPooling('pool1', shape=3, padding='VALID', stride=2)
                .Conv2D("convs4", 80, kernel_shape=1, padding="VALID")
                .Conv2D("convs5", 192, padding="VALID")
                .Conv2D("convs6", 256, stride=2, padding="VALID")
                .apply(block35repeat)
                .apply(reduction_a, 192, 192, 256, 384)
                .apply(block17repeat)
                .apply(reduction_b)
                .apply(block8repeat)
                .apply(block8, activation_fn=None)
                .apply(avg_pooling)())

            s_net = (LinearWrap(net)
                    .FullyConnected("fc1", out_dim=cfg.num_class, nl=tf.identity)())
        # softmax-loss
        # net = tf.reshape(net, [-1])
        feature = tf.identity(net, name="FEATURE")

        softmax_result = tf.nn.softmax(s_net)
        softmax_result = tf.identity(softmax_result, name="PRO")

        result_label = tf.reshape(s_net, (-1,cfg.num_class))
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_label, labels=label)
        softmax_loss = tf.reduce_sum(softmax_loss)
        # pdb.set_trace()
        softmax_loss = tf.reduce_mean(softmax_loss, name="softmax_loss")
        # loss = tf.reduce_mean(softmax_loss, name="loss")

        #tensorboard
        # wrong = symbolic_functions.prediction_incorrect(result_label, label, name='inner')
        # train_error = tf.reduce_mean(wrong, name='train_error')
        # summary.add_moving_summary(train_error)

        # center-loss1
        centers = tf.get_variable('centers', [cfg.num_class, net.get_shape()[1]], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1]) # from multi dim to 1 dim  
        centers_batch = tf.gather(centers, label)

        diff = cfg.center_loss_lr * (centers_batch - net)
        centers = tf.scatter_sub(centers, label, diff)
        center_loss = tf.reduce_mean(tf.square(net - centers_batch), name="center_loss")
        
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

def get_data(train_or_test, is_square=False):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, is_square)
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
    dataset_train = get_data('train', args.is_square == True)
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
                                     [(0, 1e-6)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        # steps_per_epoch=1280000 / int(args.batch_size),
        max_epoch=10000000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--is_square', action='store_true', help='true for face square')
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
