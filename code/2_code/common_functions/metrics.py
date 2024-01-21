import tensorflow as tf
import numpy as np
import sklearn.metrics as sk_metrics

'''
metric_group_idx=
0: indices of AUC@top_k writen by myself with TensorFlow in row-wise manner;
1: indices of AUC, AUPR, Precision, Recall and MCC calculated with sk-learn in row-wise manner;
2: indices of AUC, AUPR, Precision, Recall and MCC calculated with sk-learn in vectorized matrix;
3: indices of AUC@top_k writen by myself with TensorFlow in vectorized matrix;
'''


class MetricUtils:
    metric_groups = [['auc5_train', 'auc5_val', 'auc5_test',
                      'auc10_train', 'auc10_val', 'auc10_test',
                      'auc20_train', 'auc20_val', 'auc20_test',
                      'auc50_train', 'auc50_val', 'auc50_test',
                      'auc100_train', 'auc100_val', 'auc100_test',
                      'auc200_train', 'auc200_val', 'auc200_test',
                      'auc500_train', 'auc500_val', 'auc500_test',
                      'auc_train', 'auc_val', 'auc_test'],
                     ['auc_train', 'auc_val', 'auc_test',
                      'aupr_train', 'aupr_val', 'aupr_test',
                      'pre_train', 'pre_val', 'pre_test',
                      'rec_train', 'rec_val', 'rec_test',
                      'f1_train', 'f1_val', 'f1_test',
                      'mcc_train', 'mcc_val', 'mcc_test'],
                     ['auc_train', 'auc_val', 'auc_test',
                      'aupr_train', 'aupr_val', 'aupr_test',
                      'pre_train', 'pre_val', 'pre_test',
                      'rec_train', 'rec_val', 'rec_test',
                      'f1_train', 'f1_val', 'f1_test',
                      'mcc_train', 'mcc_val', 'mcc_test'],
                     ['auc5_train', 'auc5_val', 'auc5_test',
                      'auc10_train', 'auc10_val', 'auc10_test',
                      'auc20_train', 'auc20_val', 'auc20_test',
                      'auc50_train', 'auc50_val', 'auc50_test',
                      'auc100_train', 'auc100_val', 'auc100_test',
                      'auc200_train', 'auc200_val', 'auc200_test',
                      'auc500_train', 'auc500_val', 'auc500_test',
                      'auc_train', 'auc_val', 'auc_test']]

    @staticmethod
    def calc_metrics(pred, truth, mask, metric_group_idx):
        if metric_group_idx == 0:
            return MetricUtils.calc_auc(pred, truth, mask)
        elif metric_group_idx == 1:
            return MetricUtils.calc_metrics_sk(pred, truth, mask)
        elif metric_group_idx == 2:
            return MetricUtils.calc_metrics_sk2(pred, truth, mask)
        elif metric_group_idx == 3:
            return MetricUtils.calc_auc2(pred, truth, mask)

    @staticmethod
    def calc_auc(pred, truth, mask, top_ns=(5, 10, 20, 50, 100, 200, 500, 10 ** 8), cv_flags=(1, 2, 3)):
        pred, truth, mask = pred.astype('float32'), truth.astype('float32'), mask.astype('float32')
        aucs = tf.zeros((len(top_ns), len(cv_flags)))
        col_num = tf.shape(pred)[1]

        for j, cv_flag in enumerate(cv_flags):
            pred_flt = pred * tf.cast((mask == cv_flag) | (truth == 0), 'float32') \
                       - (1 - tf.cast((mask == cv_flag) | (truth == 0), 'float32'))
            truth_flt = truth * tf.cast(mask == cv_flag, 'float32')
            sorted_idxs = tf.argsort(pred_flt, direction='DESCENDING', axis=1)
            truth_flt = tf.gather(truth_flt, sorted_idxs, axis=1, batch_dims=1)
            good_row_flags = tf.cast(tf.reduce_sum(truth_flt, axis=1) > 0, 'float32')
            good_row_num = tf.reduce_sum(good_row_flags)

            for k, top_n in enumerate(top_ns):
                if top_n > col_num:
                    top_n = col_num

                pos_num = tf.math.count_nonzero(truth_flt, axis=1).astype('float32')
                neg_num = (col_num - pos_num).astype('float32')
                weight = tf.range(col_num, col_num - top_n, -1, 'float32').reshape((1, -1))
                s = tf.reduce_sum(truth_flt[:, :top_n] * weight, axis=1)
                auc = (s - (pos_num + 1) * pos_num / 2) / (pos_num * neg_num)
                zero_idxs = tf.where(pos_num == 0)
                one_idxs = tf.where(neg_num == 0)
                auc = tf.tensor_scatter_nd_update(auc, zero_idxs, tf.zeros(tf.shape(zero_idxs)[0]))
                auc = tf.tensor_scatter_nd_update(auc, one_idxs, tf.ones(tf.shape(one_idxs)[0]))
                auc = (auc * top_n / col_num).astype('float32')
                auc = tf.reduce_sum(auc * good_row_flags) / good_row_num
                aucs = tf.tensor_scatter_nd_update(aucs, ((k, j),), auc.reshape((1,)))

        return aucs.reshape((-1,))

    @staticmethod
    def calc_auc2(pred, truth, mask, top_ns=(5, 10, 20, 50, 100, 200, 500, 10 ** 8), cv_flags=(1, 2, 3)):
        pred, truth, mask = pred.astype('float32'), truth.astype('float32'), mask.astype('float32')
        pred, truth, mask = pred.reshape((-1,)), truth.reshape((-1,)), mask.reshape((-1,))
        aucs = tf.zeros((len(top_ns), len(cv_flags)))

        for j, cv_flag in enumerate(cv_flags):
            pred_flt = pred[mask == cv_flag]
            truth_flt = truth[mask == cv_flag]
            sorted_idxs = tf.argsort(pred_flt, direction='DESCENDING')
            truth_flt = tf.gather(truth_flt, sorted_idxs)
            inst_num = tf.shape(pred_flt)[0]

            for k, top_n in enumerate(top_ns):
                if top_n > inst_num:
                    top_n = inst_num

                pos_num = tf.math.count_nonzero(truth_flt[:top_n]).astype('float32')
                neg_num = (top_n - pos_num).astype('float32')
                weight = tf.range(top_n, 0, -1, 'float32')
                s = tf.reduce_sum(truth_flt[:top_n] * weight)
                if pos_num == 0:
                    auc = tf.zeros(())
                elif neg_num == 0:
                    auc = tf.ones(())
                else:
                    auc = (s - (pos_num + 1) * pos_num / 2) / (pos_num * neg_num)
                # auc = auc * top_n / inst_num
                aucs = tf.tensor_scatter_nd_update(aucs, ((k, j),),
                                                   (auc * top_n / inst_num).astype('float32').reshape((1,)))

        return aucs.reshape((-1,))

    @staticmethod
    def calc_metrics_sk(pred, truth, mask):
        pred = np.asarray(pred)
        truth = np.asarray(truth)
        mask = np.asarray(mask)
        pred = pred + 0
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0
        r_num = truth.shape[0]

        auc_train = 0
        auc_val = 0
        auc_test = 0
        aupr_train = 0
        aupr_val = 0
        aupr_test = 0
        pre_train = 0
        pre_val = 0
        pre_test = 0
        rec_train = 0
        rec_val = 0
        rec_test = 0
        f1_train = 0
        f1_val = 0
        f1_test = 0
        mcc_train = 0
        mcc_val = 0
        mcc_test = 0

        bad_rows = 0
        for i in range(r_num):
            row_truth_train = truth[i, mask[i, :] == 1]
            row_pred_train = pred[i, mask[i, :] == 1]
            if row_truth_train.shape[0] < 2:
                bad_rows = bad_rows + 1
                continue
            else:
                mean = np.mean(row_truth_train)
                if mean == 0 or mean == 1:
                    bad_rows = bad_rows + 1
                    continue
            gather_idxs = np.argsort(row_pred_train)[-1::-1]
            row_truth_train_gathered = np.take_along_axis(row_truth_train, gather_idxs, 0)
            non_zeros = np.count_nonzero(row_truth_train)
            row_pred_train_binary = np.zeros(row_pred_train.shape)
            row_pred_train_binary[:non_zeros] = 1

            auc_train = auc_train + sk_metrics.roc_auc_score(row_truth_train, row_pred_train)
            aupr_train = aupr_train + sk_metrics.average_precision_score(row_truth_train,
                                                                         row_pred_train)

            pre_train = pre_train + sk_metrics.precision_score(row_truth_train_gathered,
                                                               row_pred_train_binary)
            rec_train = rec_train + sk_metrics.recall_score(row_truth_train_gathered,
                                                            row_pred_train_binary)
            f1_train = f1_train + sk_metrics.f1_score(row_truth_train_gathered, row_pred_train_binary)
            mcc_train = mcc_train + sk_metrics.matthews_corrcoef(row_truth_train_gathered,
                                                                 row_pred_train_binary)

        if not r_num == bad_rows:
            auc_train = auc_train / (r_num - bad_rows)
            aupr_train = aupr_train / (r_num - bad_rows)
            pre_train = pre_train / (r_num - bad_rows)
            rec_train = rec_train / (r_num - bad_rows)
            f1_train = f1_train / (r_num - bad_rows)
            mcc_train = mcc_train / (r_num - bad_rows)

        bad_rows = 0
        for i in range(r_num):
            row_truth_val = truth[i, mask[i, :] == 2]
            row_pred_val = pred[i, mask[i, :] == 2]
            if row_truth_val.shape[0] < 2:
                bad_rows = bad_rows + 1
                continue
            else:
                mean = np.mean(row_truth_val)
                if mean == 0 or mean == 1:
                    bad_rows = bad_rows + 1
                    continue
            gather_idxs = np.argsort(row_pred_val)[-1::-1]
            row_truth_val_gathered = np.take_along_axis(row_truth_val, gather_idxs, 0)
            non_zeros = np.count_nonzero(row_truth_val)
            row_pred_val_binary = np.zeros(row_pred_val.shape)
            row_pred_val_binary[:non_zeros] = 1

            auc_val = auc_val + sk_metrics.roc_auc_score(row_truth_val, row_pred_val)
            aupr_val = aupr_val + sk_metrics.average_precision_score(row_truth_val, row_pred_val)

            pre_val = pre_val + sk_metrics.precision_score(row_truth_val_gathered, row_pred_val_binary)
            rec_val = rec_val + sk_metrics.recall_score(row_truth_val_gathered, row_pred_val_binary)
            f1_val = f1_val + sk_metrics.f1_score(row_truth_val_gathered, row_pred_val_binary)
            mcc_val = mcc_val + sk_metrics.matthews_corrcoef(row_truth_val_gathered,
                                                             row_pred_val_binary)

        if not r_num == bad_rows:
            auc_val = auc_val / (r_num - bad_rows)
            aupr_val = aupr_val / (r_num - bad_rows)
            pre_val = pre_val / (r_num - bad_rows)
            rec_val = rec_val / (r_num - bad_rows)
            f1_val = f1_val / (r_num - bad_rows)
            mcc_val = mcc_val / (r_num - bad_rows)

        bad_rows = 0
        for i in range(r_num):
            row_truth_test = truth[i, mask[i, :] == 3]
            row_pred_test = pred[i, mask[i, :] == 3]
            if row_truth_test.shape[0] < 2:
                bad_rows = bad_rows + 1
                continue
            else:
                mean = np.mean(row_truth_test)
                if mean == 0 or mean == 1:
                    bad_rows = bad_rows + 1
                    continue
            gather_idxs = np.argsort(row_pred_test)[-1::-1]
            row_truth_test_gathered = np.take_along_axis(row_truth_test, gather_idxs, 0)
            non_zeros = np.count_nonzero(row_truth_test)
            row_pred_test_binary = np.zeros(row_pred_test.shape)
            row_pred_test_binary[:non_zeros] = 1

            auc_test = auc_test + sk_metrics.roc_auc_score(row_truth_test, row_pred_test)
            aupr_test = aupr_test + sk_metrics.average_precision_score(row_truth_test, row_pred_test)

            pre_test = pre_test + sk_metrics.precision_score(row_truth_test_gathered,
                                                             row_pred_test_binary)
            rec_test = rec_test + sk_metrics.recall_score(row_truth_test_gathered,
                                                          row_pred_test_binary)
            f1_test = f1_test + sk_metrics.f1_score(row_truth_test_gathered, row_pred_test_binary)
            mcc_test = mcc_test + sk_metrics.matthews_corrcoef(row_truth_test_gathered,
                                                               row_pred_test_binary)

        if not r_num == bad_rows:
            auc_test = auc_test / (r_num - bad_rows)
            aupr_test = aupr_test / (r_num - bad_rows)
            pre_test = pre_test / (r_num - bad_rows)
            rec_test = rec_test / (r_num - bad_rows)
            f1_test = f1_test / (r_num - bad_rows)
            mcc_test = mcc_test / (r_num - bad_rows)

        return auc_train, auc_val, auc_test, aupr_train, aupr_val, aupr_test, pre_train, pre_val, pre_test, \
            rec_train, rec_val, rec_test, f1_train, f1_val, f1_test, mcc_train, mcc_val, mcc_test

    @staticmethod
    def calc_metrics_sk2(pred, truth, mask):
        pred = np.ravel(pred)
        truth = np.ravel(truth)
        mask = np.ravel(mask)
        pred = pred + 0
        pred[np.isnan(pred)] = 0
        pred[np.isinf(pred)] = 0

        truth_train = truth[mask == 1]
        if np.mean(truth_train) == 0 or np.mean(truth_train) == 1 or truth_train.shape[0] < 2:
            auc_train = 0
            aupr_train = 0
            pre_train = 0
            rec_train = 0
            f1_train = 0
            mcc_train = 0
        else:
            pred_train = pred[mask == 1]
            gather_idxs = np.argsort(pred_train)[-1::-1]
            truth_train_gathered = np.take_along_axis(truth_train, gather_idxs, 0)
            non_zeros = np.count_nonzero(truth_train)
            pred_train_binary = np.zeros(pred_train.shape)
            pred_train_binary[:non_zeros] = 1

            auc_train = sk_metrics.roc_auc_score(truth_train, pred_train)
            aupr_train = sk_metrics.average_precision_score(truth_train, pred_train)
            pre_train = sk_metrics.precision_score(truth_train_gathered, pred_train_binary)
            rec_train = sk_metrics.recall_score(truth_train_gathered, pred_train_binary)
            f1_train = sk_metrics.f1_score(truth_train_gathered, pred_train_binary)
            mcc_train = sk_metrics.matthews_corrcoef(truth_train_gathered, pred_train_binary)

        truth_val = truth[mask == 2]
        if np.mean(truth_val) == 0 or np.mean(truth_val) == 1 or truth_val.shape[0] < 2:
            auc_val = 0
            aupr_val = 0
            pre_val = 0
            rec_val = 0
            f1_val = 0
            mcc_val = 0
        else:
            pred_val = pred[mask == 2]
            gather_idxs = np.argsort(pred_val)[-1::-1]
            truth_val_gathered = np.take_along_axis(truth_val, gather_idxs, 0)
            non_zeros = np.count_nonzero(truth_val)
            pred_val_binary = np.zeros(pred_val.shape)
            pred_val_binary[:non_zeros] = 1

            auc_val = sk_metrics.roc_auc_score(truth_val, pred_val)
            aupr_val = sk_metrics.average_precision_score(truth_val, pred_val)
            pre_val = sk_metrics.precision_score(truth_val_gathered, pred_val_binary)
            rec_val = sk_metrics.recall_score(truth_val_gathered, pred_val_binary)
            f1_val = sk_metrics.f1_score(truth_val_gathered, pred_val_binary)
            mcc_val = sk_metrics.matthews_corrcoef(truth_val_gathered, pred_val_binary)

        truth_test = truth[mask == 3]
        if np.mean(truth_test) == 0 or np.mean(truth_test) == 1 or truth_test.shape[0] < 2:
            auc_test = 0
            aupr_test = 0
            pre_test = 0
            rec_test = 0
            f1_test = 0
            mcc_test = 0
        else:
            pred_test = pred[mask == 3]
            gather_idxs = np.argsort(pred_test)[-1::-1]
            truth_test_gathered = np.take_along_axis(truth_test, gather_idxs, 0)
            non_zeros = np.count_nonzero(truth_test)
            pred_test_binary = np.zeros(pred_test.shape)
            pred_test_binary[:non_zeros] = 1

            auc_test = sk_metrics.roc_auc_score(truth_test, pred_test)
            aupr_test = sk_metrics.average_precision_score(truth_test, pred_test)
            pre_test = sk_metrics.precision_score(truth_test_gathered, pred_test_binary)
            rec_test = sk_metrics.recall_score(truth_test_gathered, pred_test_binary)
            f1_test = sk_metrics.f1_score(truth_test_gathered, pred_test_binary)
            mcc_test = sk_metrics.matthews_corrcoef(truth_test_gathered, pred_test_binary)

        return auc_train, auc_val, auc_test, aupr_train, aupr_val, aupr_test, pre_train, pre_val, pre_test, \
            rec_train, rec_val, rec_test, f1_train, f1_val, f1_test, mcc_train, mcc_val, mcc_test

