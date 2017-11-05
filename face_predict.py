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

def PCA(data, K):
    # 数据标准化
    m = mean(data, axis=0) # 每列均值
    data -= m
    # 协方差矩阵
    C = cov(transpose(data))
    # 计算特征值特征向量，按降序排序
    evals, evecs = linalg.eig(C)
    indices = argsort(evals) # 返回从小到大的索引值
    indices = indices[::-1] # 反转

    evals = evals[indices] # 特征值从大到小排列
    evecs = evecs[:, indices] # 排列对应特征向量
    evecs_K_max = evecs[:, :K] # 取最大的前K个特征值对应的特征向量

    # 产生新的数据矩阵
    finaldata = dot(data, evecs_K_max)
    return finaldata

def predict_image_pairs(image1, image2, predict_func, args):
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
 
    cos_dis = 1 - pw.pairwise_distances(np.array([dis1]), np.array([dis2]), metric='cosine')
    return cos_dis

def seek_threshold(rec):
    rec = np.array(rec, dtype=np.float32).reshape(-1,2)

    index1 = np.where(rec[:,1] == 0)
    index2 = np.where(rec[:,1] == 1)

    pos = rec[index1]
    neg = rec[index2]
    max_corr_num = 0
    # for i in range(0.2, 1, 0.001):
    th = 0.01
    while th < 1:
        # print(count)
        count = 0
        pos_idx = np.where(pos[:,0] >= th)
        neg_idx = np.where(neg[:,0] < th)

        pos_corr_num = len(pos_idx[0])
        neg_corr_num = len(neg_idx[0])

        corr_num = pos_corr_num + neg_corr_num

        # print(str(th) + " acuracy: " + str(count) +  " ," +str(float(count / 6000)))
        if corr_num > max_corr_num:
            max_corr_num = corr_num
            print(max_corr_num)
            print(str(th) + " acuracy: " + str(float(max_corr_num / len(rec))) + " " + str(len(rec)) + " pos-accuracy: " + str(pos_corr_num / pos.shape[0]) + " neg-accuracy: " + str(neg_corr_num / neg.shape[0]))

        th += 0.01


def train_validation(rec):
    rec = np.array(rec, dtype=np.float32).reshape(-1,2)

    index1 = np.where(rec[:,1] == 0)
    index2 = np.where(rec[:,1] == 1)

    pos = rec[index1]
    neg = rec[index2]
    max_corr_num = 0
  
    th = 0.01
    while th < 1:
      
        count = 0
        pos_idx = np.where(pos[:,0] >= th)
        neg_idx = np.where(neg[:,0] < th)

        pos_corr_num = len(pos_idx[0])
        neg_corr_num = len(neg_idx[0])

        corr_num = pos_corr_num + neg_corr_num

        if corr_num > max_corr_num:
            max_corr_num = corr_num
            print(max_corr_num)
            print(str(th) + " acuracy: " + str(float(max_corr_num / len(rec))) + " " + str(len(rec)) + " pos-accuracy: " + str(pos_corr_num / pos.shape[0]) + " neg-accuracy: " + str(neg_corr_num / neg.shape[0]))

        th += 0.01

def predict_imags_train_dataset(pos_path, neg_path, predict_func, args):
    with open(pos_path, 'r') as f:
        pos_file = f.readlines()
        print(len(pos_file))

    with open(neg_path, 'r') as f:
        neg_file = f.readlines()
        print(len(neg_file))

   
    flage = []
    is_neg = 0
    for file in [pos_file, neg_file]:
        for item in file:
            imgs_pairs = item.strip().split(' ')
            # image1 = os.path.join(imgs_path, imgs_pairs[0] + ".jpg")
            # image2 = os.path.join(imgs_path, imgs_pairs[1] + ".jpg")
            print(imgs_pairs[0])
            cos_result = predict_image_pairs(imgs_pairs[0], imgs_pairs[1], predict_func, args)[0][0]

            print(cos_result)
            flage.append(cos_result)
            flage.append(is_neg)
        is_neg = 1
   
    train_validation(flage)


def recognition_person(img_path, predict_func, args):

    with open("dataset/person_list.txt", "r") as f:
        file = f.readlines()
    f.close()
    feature =  open("dataset/person_feature.json", "w")
    for item in enumerate(file):
        items = item.strip().split(" ")
        img_path = items[0]
        if args.flage:
            h = 224
            w = 224
        else:
            h = 160
            w = 160
        img1 = misc.imread(img_path, mode='RGB')
        img11 = np.fliplr(img1)

        img1 = cv2.resize(img1, (w, h))
        img11 = cv2.resize(img11, (w, h))

        imgs = []
        imgs.append(img1)
        imgs.append(img11)

        predict_result = predict_func([imgs])[0]
        dis1 = np.hstack((predict_result[0,:], predict_result[1,:]))
        feature.dump({items[0]:dis1})
    feature.close()

def predict_imags(pos_path, neg_path, predict_func, args):
    with open(pos_path, 'r') as f:
        pos_file = f.readlines()
        print(len(pos_file))
    with open(neg_path, 'r') as f:
        neg_file = f.readlines()
        print(len(neg_file))

    imgs_path = 'dataset/alignLfwImage'
    cos_results = []
    is_neg = 0
    for file in [pos_file, neg_file]:
        for item in file:
            imgs_pairs = item.strip().split(' ')
            image1 = os.path.join(imgs_path, imgs_pairs[0] + ".jpg")
            image2 = os.path.join(imgs_path, imgs_pairs[1] + ".jpg")
            print(image2)
            cos_result = predict_image_pairs(image1, image2, predict_func, args)[0][0]
            print(cos_result)
            cos_results.append(cos_result)
            cos_results.append(is_neg)
        is_neg = 1
    seek_threshold(cos_results)


def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)

    if args.flage:
        model = ResnetModel(args.depth)
    else:
        model = InceptionModel()

    predict_config = PredictConfig(session_init = sess_init,
                                    model = model,
                                    input_names = ["input"],
                                    output_names = ["FEATURE"])
    predict_func = OfflinePredictor(predict_config)
    return predict_func


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='train_log/face_train1023-120224/model-475431')
    parser.add_argument('--depth', default='18', type=int, choices=[18, 34, 50, 101])
    parser.add_argument('--flage', action="store_true", help="resnet")

     #align by five landmark
     # parser.add_argument('--model_path', default='train_log/face_train/model-439699')

    parser.add_argument('--input_image1')
    parser.add_argument('--input_image2', default='5.jpg')

    parser.add_argument('--input_pos_path', default='dataset/POSTIVE_PAIR.txt')
    parser.add_argument('--input_neg_path', default='dataset/NEGATIVE_PAIR.txt')

    parser.add_argument('--test_input_pos_path')
    parser.add_argument('--test_input_neg_path', default='dataset/train_test_on_webface_neg_test.txt')

    parser.add_argument('--test', action="store_true", help="test_json")
    parser.add_argument('--test_image', default="test_image.jpg", help="test image path")
    args = parser.parse_args()
   
    # if os.path.isdir("output"):
    #     shutil.rmtree("output")
    # os.mkdir('output')
 

    predict_func = get_pred_func(args)

    if args.input_image1 != None:
        predict_image_pairs(args.input_image1, args.input_image2, predict_func, args)
    elif args.test_input_pos_path != None:
        predict_imags_train_dataset(args.test_input_pos_path, args.test_input_neg_path, predict_func, args)
    elif args.test:
        recognition_person(args.test_image, predict_func, args)
    else:
        predict_imags(args.input_pos_path, args.input_neg_path, predict_func, args)

    

