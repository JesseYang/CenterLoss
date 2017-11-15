import numpy as np
from easydict import EasyDict as edict

cfg = edict()



cfg.nrof_classes = 10575#num of class label

cfg.train_list = "train_files.txt"

cfg.image_size = 224

cfg.center_loss_alfa = 0.9#center loss's learning rate

cfg. center_loss_factor = 1e-2#center loss's weight

cfg.weight_decay = 5e-5

cfg.random_crop = False

cfg.random_flip = True

cfg.shuffle = True

cfg.keep_probability = 0.8#drop out, keep rate

cfg.feature_length = 128#last full connected layer output for inception_resnet_v1

cfg.validate = True# if validate on lfw else cfg.validate=False while train model

