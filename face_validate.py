#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import ntpath
import numpy as np
import math
from scipy import misc
import argparse
import json
import cv2
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn.metrics.pairwise as pw

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug
import uuid
import tensorflow as tf
from tensorpack import *

from reader import *
from face_train_resnet import Model as ResnetModel
# from reader_face_rec import *
from face_train_inception import Model as InceptionModel
##resnet


from cfgs.config import cfg



def predict_image(image1, image2, predict_func, args):
    if args.flage:
        h = 224
        w = 224
    else:
        h = 160
        w = 160
    img1 = misc.imread(image1, mode='RGB')
    img11 = np.fliplr(img1)

    img1 = cv2.resize(img1, (w, h))
    img11 = cv2.resize(img11, (w, h))

    img2 = misc.imread(image2, mode='RGB')
    img22 = np.fliplr(img2)

    img2 = cv2.resize(img2, (w, h))
    img22 = cv2.resize(img22, (w, h))
    
    imgs = []
    imgs.append(img1)
    imgs.append(img11)
    imgs.append(img2)
    imgs.append(img22)
    predict_result = predict_func([imgs])[0]
    dis1 = np.hstack((predict_result[0,:], predict_result[1,:]))
    dis2 = np.hstack((predict_result[2,:], predict_result[3,:]))
    return dis1, dis2

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    # assert embeddings1.shape[0] == embeddings2.shape[0]
    # assert embeddings1.shape[1] == embeddings1.shape[1]

    k_fold = KFold(n_splits=nrof_folds, shuffle=True)
    indices = np.arange(len(actual_issame))

    tprs = np.zeros((nrof_folds, len(thresholds)))
    fprs = np.zeros((nrof_folds, len(thresholds)))
    accuracy = np.zeros(nrof_folds)
    import pdb
    # pdb.set_trace()
    diff = np.square(np.subtract(embeddings1, embeddings2))
    diff = np.sum(diff, 1)
    # diff = tf.square(tf.subtract(embeddings1, embeddings2))
    # diff = tf.reduce_sum(diff, 1)

    f = open("feature_no_normalize", "wb")
    pickle.dump(diff, f)
    print("ok")
    # return 
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print("step:", fold_idx)
        acc_train = np.zeros((len(thresholds)))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, diff[train_set], actual_issame[train_set])
        best_threshold = np.argmax(acc_train)
        print("max accuracy on train: ", acc_train[best_threshold])
        
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, diff[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold], diff[test_set], actual_issame[test_set])
        print("accuracy on test", accuracy[fold_idx])
        print("best thresholds", thresholds[best_threshold])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    print(accuracy)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dis, actual_issame):
    predicted_result = np.less(dis, threshold)

    tp = np.sum(np.logical_and(predicted_result, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predicted_result), np.logical_not(actual_issame)))
    fp = np.sum(np.logical_and(np.logical_not(actual_issame), predicted_result))
    fn = np.sum(np.logical_and(actual_issame, np.logical_not(predicted_result)))

    tpr = 0 if (tp + fn) == 0 else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn) == 0 else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dis.size
    return tpr, fpr, acc