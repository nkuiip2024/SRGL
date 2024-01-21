import os

from common_functions.configs import Configs
from common_functions.metrics import MetricUtils
from common_functions.utils import Utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
seed = 0

dataset = 'DrugTarget'
method = 'CGRN'
if method in {'GAT', 'RWRH'}:
    batch_size = [1500, 1500]
else:
    batch_size = [5000, 5000]
if dataset in {'DrugTarget', 'movielens_100k'} or dataset.__contains__('toy'):
    batch_sampler = 'None'
else:
    batch_sampler = 'Grid'
use_d_features = False
use_t_features = False
use_D = True
use_T = True
levels = 2
if method in {'GAT'}:
    units = 8
else:
    units = 200
optimizer = 'rmsprop'
loss_mode = 'RDT'
epochs = 200
metric_group_idx = 1
cv_metric = 'auc_val'
best_per_epoch = False

load_saved_results = True
data_config = {
    'val_prop': 0.2,
    'test_prop': 0.2,
    'balance_01': True,

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
    'method': method,
    'use_d_features': use_d_features,
    'use_t_features': use_t_features,
    'use_D': use_D,
    'use_T': use_T,
    'epochs': epochs,
    'levels': levels,
    'units': units,
    'loss_mode': loss_mode,
    'optimizer': optimizer,
    'input_feature_dim': 200,
    'seed': seed
}
if dataset.__contains__('toy'):
    model_config['input_feature_dim'] = None
if method == 'CGRN':
    CGRN_config = {
        'drop_mode': 'R',
        'drop_rate': 0,
        'GL_mode': 'R',
        'top_k': 10,
        'GC_mode': 'RDT',
        'GC_layer_alg': 'CGC',
        'MI_neg_sampler': 'FE',
        'MI_alg': 10,
        'loss_global_mode': 'None',
        'loss_MI_mode': 'R',
        'loss_LP_mode': loss_mode,
        'loss_LP_alg': 'MSE'
    }
    model_config = dict(model_config, **CGRN_config)
metric_names = MetricUtils.metric_groups[metric_group_idx]
metric_config = {
    'metric_group_idx': metric_group_idx,
    'metric_names': metric_names,
    'best_for_all_metrics': False,
    'cv_metric': cv_metric,
    'cv_metric_idx': metric_names.index(cv_metric),
    'best_per_epoch': best_per_epoch,
    'patience': 500
}

Configs.init_configs(dataset, method, data_config, batch_config, model_config, metric_config)
result = Utils.display_result(Configs.result_dir, Configs.result_file_prefix,
                              timestamp=None, display_mat=False)
_ = 1
