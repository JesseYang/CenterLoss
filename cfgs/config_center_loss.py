import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'dataset/center_loss_'

cfg.num_class = 7572

cfg.train_list = ["dataset/casia_cropped_20item1_test_affine_landmark_all_dataset_cropped1-5_all.txt"]
cfg.test_list = ["dataset/casia_cropped_20item1_test_affine_landmark_all_dataset_cropped1-5_all.txt"]


#cfg.train_list = [cfg.name + "_train_test_100.txt"]
#cfg.test_list = cfg.name + "_val.txt"

cfg.img_w = 224
cfg.img_h = 224

cfg.center_loss_lr = 0.1

cfg.center_loss_weight = 0.03

cfg.weight_decay = 1e-3

cfg.affine = False
