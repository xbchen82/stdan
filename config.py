args = {}
import random
import numpy as np
import torch as t
import model5f_mult as model
args['path'] = 'checkponint/true_1/'
# -------------------------------------------------------------------------
# 参数设置
seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

learning_rate = 0.0005
dataset = "ngsim"  # highd ngsim

args['num_worker'] = 8
args['device'] = device
args['lstm_encoder_size'] = 64
args['n_head'] = 4
args['att_out'] = 48
args['in_length'] = 16
args['out_length'] = 25
args['f_length'] = 5
args['traj_linear_hidden'] = 32
args['batch_size'] = 128
args['use_elu'] = True
args['dropout'] = 0
args['relu'] = 0.1
args['lat_length'] = 3
args['lon_length'] = 3  # 2
args['use_true_man'] = False
args['epoch'] = 20
args['use_spatial'] = False


# 多模态
args['use_maneuvers'] = False
# 单模态是否拼接预测意图
args['cat_pred'] = True
args['use_mse'] = False
args['pre_epoch'] = 6
args['val_use_mse'] = True


# -------------------------------------------------------------------------
