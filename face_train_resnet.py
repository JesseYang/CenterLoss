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
import shutil
import lfw
from reader_facenet import *
from cfgs.config import cfg
import facenet
class Model(ModelDesc):

    def __init__(self, depth=101):
        super(Model, self).__init__()
        self.depth = depth

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.image_size, cfg.image_size, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]
    def _build_graph(self, inputs):
        # with tf.device('/gpu:0'):
        image, label = inputs
       
        image = tf.identity(image, name="NETWORK_INPUT")
        tf.summary.image('input-image', image, max_outputs=5)

        image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), image)

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
                    # .FullyConnected("fc1", out_dim=1024 )
                    .FullyConnected("fc2", out_dim=1024, nl=tf.identity)())

            s_net = (LinearWrap(logits)
                    .FullyConnected("fc3", out_dim=cfg.nrof_classes, W_init = tf.truncated_normal_initializer(stddev=0.1), nl=tf.identity)())
        
        # logits = tf.sigmoid(logits) - 0.5
        embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embeddings')
        feature = tf.identity(embeddings, name='FEATURE')

            
        # softmax-loss
        # result_label = tf.reshape(s_net, (-1,cfg.num_class))
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s_net, labels=label, name="softmax_loss")
        softmax_loss = tf.reduce_mean(softmax_loss, name="softmax_loss")
       
        # center-loss
        if cfg.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(logits, label, cfg.center_loss_alfa, cfg.nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * cfg.center_loss_factor)

        center_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # total loss
        loss = tf.add_n([softmax_loss] + center_loss, name="loss")
        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            add_moving_summary(loss, wd_cost)
            add_moving_summary(softmax_loss)
            self.cost = tf.add_n([loss, wd_cost], name='cost')
        else:
            
            add_moving_summary(softmax_loss)
            self.cost = tf.identity(loss, name='cost')


    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        # return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        return tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)


class EvalLFW(Inferencer):
    def __init__(self):
        self.names = ["FEATURE"]
    
    def _get_fetches(self):
        return self.names

    def _before_inference(self):
        self.results = []
        self.img_embeddings =[]
    def _on_fetches(self, output):
        self.results.append(output)


    def _after_inference(self):
        # pdb.set_trace()
        # print(len(self.results))
        for i in range(len(self.results)):
            # self.total_features.append()
            for j in range(len(self.results[i][0])):
                self.img_embeddings.append(self.results[i][0][j])
        # pdb.set_trace()
        # print(len(self.total_features))
        pos = [True] * 3000
        neg = [False] * 3000
        actual_issame = pos + neg
        # print(np.asarray(self.img_embeddings).shape)
        # print(np.asarray(self.img_embeddings)[0])
        _, _, accuracy, best_threshold = lfw.evaluate(np.asarray(self.img_embeddings), actual_issame, nrof_folds=10)
        print('Best threshold array: ', best_threshold)
        print('Each accuracy: ', accuracy)
        print('Mean thrshold: %2.4f' %  np.mean(best_threshold))
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
        # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        return { "LFW train acc": np.mean(accuracy)}


def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, is_train = isTrain, shuffle=cfg.shuffle)
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
  
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, 50, remainder=not isTrain)
    return ds
def get_config(args):

    dataset_train = get_data('train')
    dataset_val = get_data('test')

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
                                       [(0, 1e-1), (65, 1e-2), (77, 1e-3), (1000, 1e-4), (2000, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val, [EvalLFW()]),
        ],
        model=Model(args.depth),
        steps_per_epoch=1000,
        max_epoch=1e+5,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--batch_size', default=90)
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=101, choices=[18, 34, 50, 101])
    parser.add_argument('--load', help='load model')
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
