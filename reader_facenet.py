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
        filter_ret.append([ele[0], int(ele[1])])
    return filter_ret

class Data(RNGDataFlow):

    def __init__(self, filename_list, shuffle=True):
        self.filename_list = filename_list

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        self.imglist = []
        for filename in filename_list:
            self.imglist.extend(get_img_list(filename))
        self.shuffle = shuffle
    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path, label = self.imglist[k]
            if not os.path.isfile(img_path):
                continue
            img = misc.imread(img_path, mode='RGB')
           
            if cfg.random_crop == True:#crop form 182 to 160
                # img = process(img)
                img_h, img_w, _ = img.shape
                assert (img_h >= cfg.image_size and img_w >= cfg.image_size),'img_h length error while reader data'
                h_range = random.randint(0, img_h - cfg.image_size)
                w_range = random.randint(0, img_w - cfg.image_size)
                img = img[h_range:h_range + cfg.image_size, w_range:w_range + cfg.image_size]

            else:
                img = cv2.resize(img, (cfg.image_size, cfg.image_size))
            
            if cfg.random_flip == True and (random.uniform(0, 1) > 0.5):#random flip
                img = np.fliplr(img)
                # misc.imsave(str(uuid.uuid4())+".jpg",img) 
            # misc.imsave(str(uuid.uuid4())+".jpg",img)
            # img = (img - np.average(img) / np.std(img))#per_image_standardization
            
            yield [img, label]

if __name__ == '__main__':
    ds = Data(cfg.train_list)
    # ds.reset_state()
    # count = 0
    # while count<1:
    #     g = ds.get_data()
    #     dp = next(g)
    #     count += 1
    # import pdb
    # pdb.set_trace()
