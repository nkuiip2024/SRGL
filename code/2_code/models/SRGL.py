import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.models import Model

from common_functions.GNN_layers import GCN_Layer
from common_functions.configs import Configs
from common_functions.metrics import MetricUtils


class SRGL_Model(Model):
    def __init__(self, input_data=None,
                 levels=2, GL_units=200, GC_units=200,
                 drop_mode='None', drop_rate=0.2, GL_mode='R', top_k=10, GC_mode='RDT',
                 GC_layer_alg='CGC', loss_global_mode='R', loss_MI_mode='R',
                 MI_neg_sampler='FE', MI_alg=1, loss_LP_mode='R', loss_LP_alg='BPR',
                 epochs=200, optimizer='adam', seed=0, hyper_paras=(1, 1, 1, 1), **kwargs):
        super(SRGL_Model, self).__init__()
        self.levels = levels
        self.GL_units = GL_units
        self.GC_units = GC_units
        if kwargs.__contains__('units'):
            self.GL_units = kwargs['units']
            self.GC_units = kwargs['units']
        self.drop_mode = drop_mode
        self.drop_rate = drop_rate
        self.GL_mode = GL_mode
        self.top_k = top_k
        self.GC_mode = GC_mode
        self.GC_layer_alg = GC_layer_alg
        self.MI_neg_sampler = MI_neg_sampler
        self.MI_alg = MI_alg
        self.loss_global_mode = loss_global_mode
        self.loss_MI_mode = loss_MI_mode
        self.loss_LP_mode = loss_LP_mode
        if kwargs.__contains__('loss_mode'):
            self.loss_LP_mode = kwargs['loss_mode']
        self.loss_LP_alg = loss_LP_alg
        self.epochs = epochs
        self._optimizer = optimizer
        self.seed = seed
        self.w_global, self.w_MI, self.w_LP_D, self.w_LP_T = hyper_paras
        self.kwargs = kwargs

        self.init_data(input_data)

        self.ED_layer = [ED_Layer(drop_rate=self.drop_rate,
                                  seed=self.seed + i)
                         for i in range(self.levels)]
        self.GL_layers = [GL_Layer(units=self.GL_units,
                                   GL_mode=self.GL_mode,
                                   top_k=self.top_k,
                                   seed=self.seed + i)
                          for i in range(self.levels)]
        if self.GC_layer_alg == 'GCN':
            self.GC_layers = [GCN_Layer(units=self.GC_units,
                                        seed=self.seed + i)
                              for i in range(self.levels)]
            self.GC_readout = GCN_Layer(units=self.GC_units,
                                        seed=self.seed)
            self.GC_output = GCN_Layer(units=self.GC_units,
                                       seed=self.seed)
        elif self.GC_layer_alg == 'CGC':
            self.GC_layers = [CGC_Layer(units=self.GC_units,
                                        GC_mode=self.GC_mode,
                                        seed=self.seed + i)
                              for i in range(self.levels + 2)]
            self.GC_readout = CGC_Layer(units=self.GC_units,
                                        GC_mode=self.GC_mode,
                                        seed=self.seed)
            self.GC_output = CGC_Layer(units=self.GC_units,
                                       GC_mode=self.GC_mode,
                                       seed=self.seed)

        self.es_callback = EarlyStopping(monitor='loss_LP', patience=self.epochs,
                                         mode='min', restore_best_weights=True)
        self.compile(optimizer=self._optimizer)

    def init_data(self, input_data):
        [D, T, R_train, R_truth, H_d, H_t, mask] = input_data[:7]
        self.R_train = tf.convert_to_tensor(R_train, dtype='float32')
        self.R_truth = tf.convert_to_tensor(R_truth, dtype='float32')
        self.mask = tf.convert_to_tensor(mask, dtype='float32')
        self.D = tf.convert_to_tensor(D, dtype='float32')
        self.T = tf.convert_to_tensor(T, dtype='float32')

        self.H_d = tf.convert_to_tensor(H_d, dtype='float32')
        self.H_t = tf.convert_to_tensor(H_t, dtype='float32')

        self.R_pred = None
        self.H_d_out = None
        self.H_t_out = None

        if self.loss_MI_mode in {'R', 'RDT'}:
            self.W_MI = self.add_weight(name='W_MI',
                                        shape=(2 * self.GC_units, 2 * self.GC_units),
                                        initializer=GlorotUniform(self.seed))

        self.dense_ec_R = self.dense_ec_D = self.dense_ec_T = None
        if self.loss_LP_alg.startswith('EC'):
            if self.loss_LP_mode.__contains__('R'):
                self.dense_ec_R = Dense(units=1,
                                        activation=tf.nn.sigmoid,
                                        use_bias=False,
                                        kernel_initializer=GlorotUniform(self.seed))
            if self.loss_LP_mode.__contains__('DT'):
                self.dense_ec_D = Dense(units=1,
                                        activation=tf.nn.sigmoid,
                                        use_bias=False,
                                        kernel_initializer=GlorotUniform(self.seed + 1))
                self.dense_ec_T = Dense(units=1,
                                        activation=tf.nn.sigmoid,
                                        use_bias=False,
                                        kernel_initializer=GlorotUniform(self.seed + 2))

    def call(self, inputs, training=None, mask=None):
        H_d_in, H_t_in = inputs[0][0], inputs[1][0]

        R_rec = R_res = R_crp = R_raw = self.R_train
        D_rec = D_res = D_crp = D_raw = self.D
        T_rec = T_res = T_crp = T_raw = self.T
        H_d_raw = H_d_rec = H_d_in
        H_t_raw = H_t_rec = H_t_in

        for i in range(self.levels):
            R_crp, D_crp, T_crp = R_rec, D_rec, T_rec
            if self.drop_mode.__contains__('R'):
                R_crp = self.ED_layer[i](R_rec)
            if self.drop_mode.__contains__('DT'):
                D_crp = self.ED_layer[i](D_rec)
                T_crp = self.ED_layer[i](T_rec)

            H_d_raw, H_t_raw = self.GC_layers[i]([R_raw, D_raw, T_raw, H_d_in, H_t_in])
            H_d_crp, H_t_crp = self.GC_layers[i]([R_crp, D_crp, T_crp, H_d_in, H_t_in])

            R_res, R_res_flt, D_res, D_res_flt, T_res, T_res_flt = self.GL_layers[i](
                [H_d_crp, H_t_crp])
            R_rec, D_rec, T_rec = R_crp, D_crp, T_crp
            if self.GL_mode.__contains__('R'):
                R_rec += R_res
            if self.GL_mode.__contains__('DT'):
                D_rec += D_res
                T_rec += T_res
            H_d_rec, H_t_rec = self.GC_layers[i]([R_rec, D_rec, T_rec, H_d_in, H_t_in])

        pos_idxs_R = neg_idxs_R = pos_idxs_D = neg_idxs_D = pos_idxs_T = neg_idxs_T = None
        if self.loss_MI_mode.__contains__('R') or self.loss_LP_mode.__contains__('R'):
            pos_idxs_R, neg_idxs_R = self.pos_neg_sampling(R_raw)
        if self.loss_MI_mode.__contains__('DT') or self.loss_LP_mode.__contains__('DT'):
            pos_idxs_D, neg_idxs_D = self.pos_neg_sampling(D_raw)
            pos_idxs_T, neg_idxs_T = self.pos_neg_sampling(T_raw)

        loss_global_R = loss_global_D = loss_global_T = 0
        if self.loss_global_mode.__contains__('R'):
            loss_global_R = self.loss_global(R_rec, R_raw)
            tf.print('loss_global_R:', loss_global_R)
        if self.loss_global_mode.__contains__('DT'):
            loss_global_D = self.loss_global(D_rec, D_raw)
            loss_global_T = self.loss_global(T_rec, T_raw)
            tf.print('loss_global_D:', loss_global_D, end=', ')
            tf.print('loss_global_T:', loss_global_T, end=', ')
        loss_global = loss_global_R + loss_global_D + loss_global_T

        loss_MI_R = loss_MI_D = loss_MI_T = 0
        if self.loss_MI_mode.__contains__('R'):
            loss_MI_R = self.loss_MI(R_rec, R_raw, self.W_MI, H_d_rec, H_t_rec, H_d_raw,
                                     H_t_raw,
                                     self.MI_neg_sampler, self.MI_alg, pos_idxs_R, neg_idxs_R)
            tf.print('loss_MI_R:', loss_MI_R, end=', ')
        if self.loss_MI_mode.__contains__('DT'):
            loss_MI_D = self.loss_MI(D_rec, D_raw, self.W_MI, H_d_rec, H_d_rec, H_d_raw,
                                     H_d_raw,
                                     self.MI_neg_sampler, self.MI_alg, pos_idxs_D, neg_idxs_D)
            loss_MI_T = self.loss_MI(T_rec, T_raw, self.W_MI, H_t_rec, H_t_rec, H_t_raw,
                                     H_t_raw,
                                     self.MI_neg_sampler, self.MI_alg, pos_idxs_T, neg_idxs_T)
            tf.print('loss_MI_D:', loss_MI_D, end=', ')
            tf.print('loss_MI_T:', loss_MI_T, end=', ')
        loss_MI = loss_MI_R + loss_MI_D + loss_MI_T

        loss_LP_R = loss_LP_D = loss_LP_T = 0
        R_pred = tf.einsum('ij,kj->ik', H_d_rec, H_t_rec)

        if self.loss_LP_mode.__contains__('R'):
            if self.loss_LP_alg == 'MSE':
                loss_LP_R = self.loss_MSE(R_pred, R_raw)
            elif self.loss_LP_alg == 'BPR':
                loss_LP_R = self.loss_BPR(R_pred, pos_idxs_R, neg_idxs_R)
            tf.print('loss_LP_R:', loss_LP_R, end=', ')
        if self.loss_LP_mode.__contains__('DT'):
            D_pred = tf.einsum('ij,kj->ik', H_d_rec, H_d_rec)
            T_pred = tf.einsum('ij,kj->ik', H_t_rec, H_t_rec)
            if self.loss_LP_alg == 'MSE':
                loss_LP_D = self.loss_MSE(D_pred, D_raw)
                loss_LP_T = self.loss_MSE(T_pred, T_raw)
            elif self.loss_LP_alg == 'BPR':
                loss_LP_D = self.loss_BPR(D_pred, pos_idxs_D, neg_idxs_D)
                loss_LP_T = self.loss_BPR(T_pred, pos_idxs_T, neg_idxs_T)
            tf.print('loss_LP_D:', loss_LP_D, end=', ')
            tf.print('loss_LP_T:', loss_LP_T, end=', ')
        tf.print()

        loss_LP = loss_LP_R + self.w_LP_D * loss_LP_D + self.w_LP_T * loss_LP_T

        self.add_metric(loss_LP, name='loss_LP', aggregation='mean')
        self.add_loss(loss_LP + self.w_global * loss_global + self.w_MI * loss_MI)

        return [tf.expand_dims(R_pred, axis=0),
                tf.expand_dims(H_d_rec, axis=0),
                tf.expand_dims(H_t_rec, axis=0)]

    @staticmethod
    def loss_global(X_res, X):
        return tf.reduce_mean(tf.square(X_res - X))

    @staticmethod
    def loss_MI(X_rec, X_raw, W, H_r_rec, H_c_rec, H_r_raw, H_c_raw, neg_sampler, MI_alg,
                pos_idxs_raw, neg_idxs_raw):
        H_r_rec_pos = tf.gather(H_r_rec, pos_idxs_raw[:, 0], axis=0)
        H_c_rec_pos = tf.gather(H_c_rec, pos_idxs_raw[:, 1], axis=0)
        H_edge_rec_pos = tf.concat([H_r_rec_pos, H_c_rec_pos], axis=1)
        h_graph_rec_pos = tf.reduce_mean(H_edge_rec_pos, axis=0)

        if neg_sampler == 'FS':
            shuf_idxs_r = tf.argsort(tf.random.uniform((H_r_rec.shape[0],)))
            shuf_idxs_c = tf.argsort(tf.random.uniform((H_c_rec.shape[0],)))
            H_r_rec_shuf = tf.gather(H_r_rec, shuf_idxs_r, axis=0)
            H_c_rec_shuf = tf.gather(H_c_rec, shuf_idxs_c, axis=0)
            H_r_rec_neg = tf.gather(H_r_rec_shuf, pos_idxs_raw[:, 0], axis=0)
            H_c_rec_neg = tf.gather(H_c_rec_shuf, pos_idxs_raw[:, 1], axis=0)
            H_edge_rec_neg = tf.concat([H_r_rec_neg, H_c_rec_neg], axis=1)
        elif neg_sampler == 'FE':
            H_r_fake = tf.gather(H_r_rec, neg_idxs_raw[:, 0], axis=0)
            H_c_fake = tf.gather(H_c_rec, neg_idxs_raw[:, 1], axis=0)
            H_edge_rec_neg = tf.concat([H_r_fake, H_c_fake], axis=1)

        H_r_raw_pos = tf.gather(H_r_raw, pos_idxs_raw[:, 0], axis=0)
        H_c_raw_pos = tf.gather(H_c_raw, pos_idxs_raw[:, 1], axis=0)
        H_edge_raw_pos = tf.concat([H_r_raw_pos, H_c_raw_pos], axis=1)
        h_graph_raw_pos = tf.reduce_mean(H_edge_raw_pos, axis=0)

        H_r_raw_neg = tf.gather(H_r_raw, neg_idxs_raw[:, 0], axis=0)
        H_c_raw_neg = tf.gather(H_c_raw, neg_idxs_raw[:, 1], axis=0)
        H_edge_raw_neg = tf.concat([H_r_raw_neg, H_c_raw_neg], axis=1)
        h_graph_raw_neg = tf.reduce_mean(H_edge_raw_neg, axis=0)

        flags = [None] * 4
        t = MI_alg
        for i in range(3, -1, -1):
            flags[i] = t % 2
            t = t // 2

        loss_MI_pos1 = loss_MI_pos2 = loss_MI_neg1 = loss_MI_neg2 = 0
        if flags[0]:
            MI_pos1 = tf.einsum('ij,jk,k->i', H_edge_rec_pos, W, h_graph_raw_pos)
            loss_MI_pos1 = -tf.reduce_mean(tf.math.log_sigmoid(MI_pos1))
        if flags[1]:
            MI_pos2 = tf.einsum('ij,jk,k->i', H_edge_raw_pos, W, h_graph_rec_pos)
            loss_MI_pos2 = -tf.reduce_mean(tf.math.log_sigmoid(MI_pos2))
        if flags[2]:
            MI_neg1 = tf.einsum('ij,jk,k->i', H_edge_rec_neg, W, h_graph_raw_pos)
            loss_MI_neg1 = -tf.reduce_mean(tf.math.log_sigmoid(1 - MI_neg1))
        if flags[3]:
            MI_neg2 = tf.einsum('ij,jk,k->i', H_edge_rec_pos, W, h_graph_raw_neg)
            loss_MI_neg2 = -tf.reduce_mean(tf.math.log_sigmoid(1 - MI_neg2))

        return loss_MI_pos1 + loss_MI_pos2 + loss_MI_neg1 + loss_MI_neg2

    @staticmethod
    def loss_MSE(X_pred, X):
        loss_pos = tf.reduce_mean(tf.square(
            tf.cast(X == 1, 'float32') * (X_pred - X)))
        loss_neg = tf.reduce_mean(tf.square(
            tf.cast(X == 0, 'float32') * (X_pred - X)))
        loss = loss_pos + 0.2 * loss_neg
        return loss

    @staticmethod
    def pos_neg_sampling_by_row(X):
        pos_idxs = tf.where(X == 1).astype('int32')
        pos_num_per_row = tf.reduce_sum((X == 1).astype('int32'), axis=1)
        neg_idxs = tf.zeros((0, 2), 'int32')
        for i in tf.range(tf.shape(X)[0]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(neg_idxs, tf.TensorShape([None, 2]))])
            zero_idxs = tf.where(X[i, :] == 0).astype('int32')
            zero_idxs = tf.random.shuffle(zero_idxs)
            pos_num = pos_num_per_row[i]
            zero_idxs = zero_idxs[:pos_num]
            _neg_idxs = tf.concat(
                [tf.ones((pos_num, 1), 'int32') * i, zero_idxs.reshape((-1, 1))], axis=1)
            neg_idxs = tf.concat([neg_idxs, _neg_idxs], axis=0)
        return pos_idxs, neg_idxs

    @staticmethod
    def pos_neg_sampling(X):
        pos_idxs = tf.where(X == 1)
        if tf.shape(pos_idxs)[0] > 100:
            r = 1.5
        else:
            r = 10.
        pos_rate = tf.reduce_mean(X)
        mask = (tf.random.uniform(X.shape) < (pos_rate * r)).astype('float32')
        neg_idxs = tf.where((1 - X) * mask > 0)
        neg_idxs = tf.random.shuffle(neg_idxs)[:tf.shape(pos_idxs)[0]]
        return pos_idxs, neg_idxs

    @staticmethod
    def loss_BPR(X_pred, pos_idxs, neg_idxs):
        X_pred_pos = tf.gather_nd(X_pred, pos_idxs)
        X_pred_neg = tf.gather_nd(X_pred, neg_idxs)
        loss = -tf.reduce_mean(tf.math.log_sigmoid(X_pred_pos - X_pred_neg))
        return loss

    def get_config(self):
        config = {
            'levels': self.levels,
            'GL_units': self.GL_units,
            'GC_units': self.GC_units,
            'drop_mode': self.drop_mode,
            'drop_rate': self.drop_rate,
            'GL_mode': self.GL_mode,
            'top_k': self.top_k,
            'GC_mode': self.GC_mode,
            'GC_layer_alg': self.GC_layer_alg,
            'MI_neg_sampler': self.MI_neg_sampler,
            'MI_alg': self.MI_alg,
            'loss_global_mode': self.loss_global_mode,
            'loss_MI_mode': self.loss_MI_mode,
            'loss_LP_mode': self.loss_LP_mode,
            'loss_LP_alg': self.loss_LP_alg,
            'epochs': self.epochs,
            'optimizer': self._optimizer,
            'seed': self.seed
        }
        return dict(config, **self.kwargs)

    def fit(self, **kwargs):
        H_d = tf.expand_dims(self.H_d, axis=0)
        H_t = tf.expand_dims(self.H_t, axis=0)
        x = [H_d, H_t]

        super().fit(x=x, batch_size=1, epochs=self.epochs, verbose=1, callbacks=[self.es_callback])
        [best_R_pred, best_H_d_out, best_H_t_out] = super().predict(x=x, batch_size=1, verbose=0)

        self.R_pred = np.squeeze(best_R_pred)
        self.R_pred[np.isnan(self.R_pred)] = 0
        self.R_pred[np.isinf(self.R_pred)] = 0
        self.H_d_out = np.squeeze(best_H_d_out)
        self.H_d_out[np.isnan(self.H_d_out)] = 0
        self.H_d_out[np.isinf(self.H_d_out)] = 0
        self.H_t_out = np.squeeze(best_H_t_out)
        self.H_t_out[np.isnan(self.H_t_out)] = 0
        self.H_t_out[np.isinf(self.H_t_out)] = 0

    def predict(self, **kwargs):
        return [self.R_pred, self.H_d_out, self.H_t_out, None]


class ED_Layer(Layer):
    def __init__(self, drop_rate=0.2, seed=0):
        super(ED_Layer, self).__init__()
        self.drop_rate = drop_rate
        self.seed = seed

    def call(self, A, **kwargs):
        mask = tf.random.uniform(A.shape, seed=self.seed)
        mask = tf.cast(mask > self.drop_rate, 'float32')
        A = A * mask
        return A


class GL_Layer(Layer):
    def __init__(self, units=200, GL_mode='R', top_k=10, seed=0):
        super(GL_Layer, self).__init__()
        self.units = units
        self.GL_mode = GL_mode
        self.top_k = top_k
        self.seed = seed

    def build(self, input_shape):
        d_dim = input_shape[0][-1]
        t_dim = input_shape[1][-1]
        self.W1 = self.add_weight(name='W1',
                                  shape=(d_dim, self.units),
                                  initializer=GlorotUniform(self.seed))
        self.W2 = self.add_weight(name='W2',
                                  shape=(t_dim, self.units),
                                  initializer=GlorotUniform(self.seed + 1))

    def call(self, inputs, **kwargs):
        H_d, H_t = inputs
        H_d = tf.matmul(H_d, self.W1)
        H_t = tf.matmul(H_t, self.W2)
        # H_d = tf.nn.l2_normalize(H_d, axis=1)
        # H_t = tf.nn.l2_normalize(H_t, axis=1)
        R_res = R_res_flt = D_res = D_res_flt = T_res = T_res_flt = tf.zeros(())
        d_num, t_num = H_d.shape[0], H_t.shape[0]
        if self.top_k > t_num:
            self.top_k = t_num

        if self.GL_mode.__contains__('R'):
            R_res = tf.nn.sigmoid(tf.einsum('ij,kj->ik', H_d, H_t))
            gather_idxs = tf.argsort(R_res, axis=1, direction='DESCENDING')
            R_res_sorted = tf.gather(R_res, gather_idxs, axis=1, batch_dims=1)
            mask = tf.concat(
                [tf.ones((d_num, self.top_k)), tf.zeros((d_num, t_num - self.top_k))], axis=1)
            R_res_sorted = R_res_sorted * mask
            gather_idxs_inv = tf.argsort(gather_idxs, axis=1)
            R_res_flt = tf.gather(R_res_sorted, gather_idxs_inv, axis=1, batch_dims=1)

        if self.GL_mode.__contains__('DT'):
            D_res = tf.nn.sigmoid(tf.einsum('ij,kj->ik', H_d, H_d))
            gather_idxs = tf.argsort(D_res, axis=1, direction='DESCENDING')
            D_res_sorted = tf.gather(D_res, gather_idxs, axis=1, batch_dims=1)
            mask = tf.concat(
                [tf.ones((d_num, self.top_k)), tf.zeros((d_num, d_num - self.top_k))], axis=1)
            D_res_sorted = D_res_sorted * mask
            gather_idxs_inv = tf.argsort(gather_idxs, axis=1)
            D_res_flt = tf.gather(D_res_sorted, gather_idxs_inv, axis=1, batch_dims=1)

            T_res = tf.nn.sigmoid(tf.einsum('ij,kj->ik', H_t, H_t))
            gather_idxs = tf.argsort(T_res, axis=1, direction='DESCENDING')
            T_res_sorted = tf.gather(T_res, gather_idxs, axis=1, batch_dims=1)
            mask = tf.concat(
                [tf.ones((t_num, self.top_k)), tf.zeros((t_num, t_num - self.top_k))], axis=1)
            T_res_sorted = T_res_sorted * mask
            gather_idxs_inv = tf.argsort(gather_idxs, axis=1)
            T_res_flt = tf.gather(T_res_sorted, gather_idxs_inv, axis=1, batch_dims=1)

        return R_res, R_res_flt, D_res, D_res_flt, T_res, T_res_flt


class CGC_Layer(Layer):
    def __init__(self, units=200, GC_mode='RDT', activation=tf.nn.relu, seed=0):
        super(CGC_Layer, self).__init__()
        self.units = units
        self.GC_mode = GC_mode
        self.activation = activation
        self.seed = seed

    def build(self, input_shapes):
        self.dense_d = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed))
        self.dense_t = Dense(units=self.units,
                             activation=self.activation,
                             use_bias=False,
                             kernel_initializer=GlorotUniform(self.seed + 1))
        self.W_d = self.add_weight(name='W_d',
                                   shape=(input_shapes[-2][-1], self.units),
                                   initializer=GlorotUniform(self.seed))
        self.W_t = self.add_weight(name='W_t',
                                   shape=(input_shapes[-1][-1], self.units),
                                   initializer=GlorotUniform(self.seed))

        self.dense_d2 = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed + 2))
        self.dense_t2 = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed + 3))
        self.dense_dd = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_dt = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_td = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))
        self.dense_tt = Dense(units=self.units,
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=GlorotUniform(self.seed))

    def call(self, inputs, **kwargs):
        R, D, T, H_d, H_t = inputs
        H_d_concat_list, H_t_concat_list = [H_d], [H_t]

        if self.GC_mode.__contains__('R'):
            n_neigh_dt = tf.reduce_sum(R, axis=1, keepdims=True)
            n_neigh_td = tf.reduce_sum(R, axis=0, keepdims=True)
            n_neigh_dt = n_neigh_dt + (n_neigh_dt == 0).astype('float32')
            n_neigh_td = n_neigh_td + (n_neigh_td == 0).astype('float32')
            n_neigh_dt_norm = n_neigh_dt ** -0.5
            n_neigh_td_norm = n_neigh_td ** -0.5
            R_norm = n_neigh_dt_norm * R * n_neigh_td_norm

            H_dt = self.activation(R_norm @ H_t)
            H_td = self.activation(R_norm.T @ H_d)
            H_d_concat_list += [H_dt]
            H_t_concat_list += [H_td]
        if self.GC_mode.__contains__('DT'):
            n_neigh_dd = tf.reduce_sum(D, axis=1, keepdims=True)
            n_neigh_tt = tf.reduce_sum(T, axis=1, keepdims=True)
            n_neigh_dd = n_neigh_dd + (n_neigh_dd == 0).astype('float32')
            n_neigh_tt = n_neigh_tt + (n_neigh_tt == 0).astype('float32')
            n_neigh_dd_norm = n_neigh_dd ** -0.5
            n_neigh_tt_norm = n_neigh_tt ** -0.5
            D_norm = n_neigh_dd_norm * D * n_neigh_dd_norm.T
            T_norm = n_neigh_tt_norm * T * n_neigh_tt_norm.T

            H_dd = self.activation(D_norm @ H_d)
            H_tt = self.activation(T_norm @ H_t)
            H_d_concat_list += [H_dd]
            H_t_concat_list += [H_tt]

        H_d_out = self.dense_d(tf.concat(H_d_concat_list, axis=1))
        H_t_out = self.dense_t(tf.concat(H_t_concat_list, axis=1))
        return H_d_out, H_t_out
