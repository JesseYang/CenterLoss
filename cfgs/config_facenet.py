import numpy as np
from easydict import EasyDict as edict

cfg = edict()



cfg.nrof_classes = 10575#num of class label

cfg.train_list = "train_files.txt"
cfg.test_list = "lfw_files.txt"


cfg.center_loss_alfa = 0.9#center loss's learning rate

cfg. center_loss_factor = 1e-2#center loss's weight

cfg.weight_decay = 5e-5



cfg.random_flip = True

cfg.shuffle = True

cfg.keep_probability = 0.8#drop out, keep rate


cfg.validate = True# if validate on lfw else cfg.validate=False while train model





#inception_resnet_v1
#cfg.image_size = 160

#resnet
cfg.image_size = 224

#inception_resnet_v1
#cfg.feature_length = 128#last full connected layer output for inception_resnet_v1
#resnet
cfg.feature_length = 1024#last full connected layer output for inception_resnet_v1


#inception_resnet_v1
# cfg.random_crop = True
#resnet
cfg.random_crop = False