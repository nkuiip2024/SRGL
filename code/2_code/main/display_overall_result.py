import os

from common_functions.configs import Configs
from common_functions.metrics import MetricUtils
from common_functions.utils import Utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
seed = 0

dataset = 'toy_dataset'
batch_sampler = 'None'
batch_size = [5000, 5000]
methods = ['CGRN', 'GIN', 'NGCF', 'CMRL', 'DGI', 'GCA', 'EGLN']
use_d_features = False
use_t_features = False
use_D = False
use_T = False
levels = 2
units = 256
optimizer = 'rmsprop'
loss_mode = 'R'
epochs = 200
metric_group_idx = 2
cv_metric = 'auc_test'
best_per_epoch = False

data_config = {
    'val_prop': 0.0,
    'test_prop': 0.3,
    'balance_01': True,
    'n_folds': 1,

    'filter_RDT': False,
    'R_threshold': 4,
    'D_threshold': 2,
    'T_threshold': 2,
    'binary_R': True,
    'binary_D': True,
    'binary_T': True,

    'extra_process': False,
    'new_d_proportion': 0.0,
    'new_t_proportion': 0.0,
    'new_d_only': False,
    'new_t_only': False,
    'd_sparsity': 0.0,
    't_sparsity': 0.0,
    'r_sparsity': 0.0,
    'r_noise_add': 0.2,
    'r_noise_del': 0.0,
    'd_noise_add': 0.0,
    'd_noise_del': 0.0,
    't_noise_add': 0.0,
    't_noise_del': 0.0
}
batch_config = {
    'batch_sampler': batch_sampler,  # None, Grid, RW, NN
    'shuffle_dt': True,  # batch_sampler==Grid
    'd_batch_size': batch_size[0],  # batch_sampler==Grid
    't_batch_size': batch_size[1],  # batch_sampler==Grid
    'min_d_num': 1000,  # batch_sampler==Grid
    'min_t_num': 1000,  # batch_sampler==Grid
    'batch_num': 100,  # batch_sampler==RW or NN
    'sample_DT': True,  # batch_sampler==RW or NN
    'path_len': 100,  # batch_sampler==RW
    'path_num': 200,  # batch_sampler==RW
    'min_node_num': 500,  # batch_sampler==RW or NN
    'max_node_num': 2500,
    'neigh_level': 3,  # batch_sampler==NN
}
model_config = {
    'use_d_features': use_d_features,
    'use_t_features': use_t_features,
    'use_D': use_D,
    'use_T': use_T,
    'epochs': epochs,
    'levels': levels,
    'units': units,
    'loss_mode': loss_mode,
    'optimizer': optimizer,
    'input_feature_dim': None,
    'seed': seed,

    # CGRN config
    'drop_mode': 'R',
    'drop_rate': 0,
    'GL_mode': 'R',
    'top_k': 10,
    'GC_mode': 'R',
    'GC_layer_alg': 'CGC',
    'MI_neg_sampler': 'FE',
    'MI_alg': 10,
    'loss_global_mode': 'None',
    'loss_MI_mode': 'R',
    'loss_LP_mode': loss_mode,
    'loss_LP_alg': 'MSE'

}
metric_names = MetricUtils.metric_groups[metric_group_idx]
metric_config = {
    'metric_group_idx': metric_group_idx,
    'metric_names': metric_names,
    'best_for_all_metrics': False,
    'cv_metric': cv_metric,
    'cv_metric_idx': metric_names.index(cv_metric),
    'best_per_epoch': best_per_epoch,
    'patience': 50
}

Utils.display_overall_result(dataset, methods, data_config, batch_config, model_config,
                             metric_config)
