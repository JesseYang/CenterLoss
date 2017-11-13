import os, sys
import pdb
import pickle
import numpy as np
from scipy import misc
import random
import six
from six.moves import urllib, range
import copy
import logging
import cv2
import uuid
from tensorpack import *
from cfgs.config import cfg

def get_img_list(text_file):
    with open(text_file) as f:
        content = f.readlines()
    ret = [record.strip().split(' ') for record in content]
    filter_ret = []
    for idx, ele in enumerate(ret):
        filter_ret.append([ele[0], int(ele[1]), ele[12:]])
    return filter_ret

class Data(RNGDataFlow):

    def __init__(self, filename_list, shuffle=True, is_square=False):
        self.filename_list = filename_list

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        self.imglist = []
        for filename in filename_list:
            self.imglist.extend(get_img_list(filename))
        self.shuffle = shuffle
        self.is_square = is_square

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path, label, coors = self.imglist[k]
            if not os.path.isfile(img_path):
                continue
            img = misc.imread(img_path, mode='RGB')  
            coors = np.array(coors, dtype=np.float32)
            coors = [int(e) for e in coors]
       
            h, w = img.shape[0:2]
            if coors[2] > w or coors[3] > h:
                continue

            xmin = coors[0]
            ymin = coors[1]
            xmax = coors[2]
            ymax = coors[3]
            if self.is_square:
                face_h = ymax - ymin
                face_w = xmax - xmin

                max_bound = np.maximum(face_h, face_w) // 2
                ori_center_x = (xmin + xmax) // 2
                ori_center_y = (ymin + ymax) // 2

                xmin = ori_center_x - max_bound
                ymin = ori_center_y - max_bound
                xmax = ori_center_x + max_bound
                ymax = ori_center_y + max_bound
            
            imgs = img[ymin:ymax, xmin:xmax]
            # misc.imsave(str(uuid.uuid4())+".jpg",imgs)
            imgs = cv2.resize(imgs, (cfg.img_w, cfg.img_h))
            yield [imgs, label]

if __name__ == '__main__':
    ds = Data(cfg.train_list)
    ds.reset_state()
    # while 1:
    #     g = ds.get_data()
    #     dp = next(g)
    # import pdb
    # pdb.set_trace()
