#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : shixiangfu
import sys
import tensorflow as tf
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
import six
###############
def dnn_model(features=None,
              params=None):
    ###############

    CONTINUOUS_COLS = params['CONTINUOUS_COLS']
    DEEP_EMBEDDING_COLS = params['DEEP_EMBEDDING_COLS']

    DEEP_SHARED_EMBEDDING_COLS = params['DEEP_SHARED_EMBEDDING_COLS']
    _HIDDEN_UNITS = params['_HIDDEN_UNITS']


    ##################
    if True:
        with variable_scope.variable_scope(
                'Deep_model',
                values=tuple(six.itervalues(features)),
        ) as scope:
            deep_sum = []
            for col, vals, embedding_size, col_type in DEEP_EMBEDDING_COLS:
                bucket_size = vals if isinstance(vals, int) else len(vals)
                embeddings = tf.get_variable(
                    shape=[bucket_size, embedding_size],
                    initializer=init_ops.glorot_uniform_initializer(),
                    name="deep_embedding_" + col
                )
                if col_type != 'int':
                    indices = string_ops.string_to_hash_bucket_fast(
                        features[col], bucket_size, name="deep_hash_" + col)
                else:
                    table = tf.contrib.lookup.index_table_from_tensor(vals)
                    indices = table.lookup(features[col])
                seq_emb = tf.nn.embedding_lookup(embeddings, indices, name="deep_lookup_" + col)
                if col_type == 'list':
                    print("test my seq:",col)
                    seq_emb = tf.reduce_mean(seq_emb,1)
                    print(seq_emb)
                deep_sum.append(seq_emb)
            for cols,vals,embedding_size,col_type,shared_flag in DEEP_SHARED_EMBEDDING_COLS:
                def get_indices(col,embedding_size,bucket_size):
                    if col_type != 'int':
                        indices = string_ops.string_to_hash_bucket_fast(
                            features[col], bucket_size, name="deep_shared_hash_" + col+str(shared_flag))
                    else:
                        table = tf.contrib.lookup.index_table_from_tensor(embedding_size)
                        indices = table.lookup(features[col])
                    return indices

                bucket_size = vals if isinstance(vals, int) else len(vals)
                embeddings = tf.get_variable(
                    shape=[bucket_size, embedding_size],
                    initializer=init_ops.glorot_uniform_initializer(),
                    name="deep_shared_embedding_" + '_'.join(c for c in cols)+str(shared_flag)
                )
                for col in cols:
                    indices = get_indices(col,embedding_size,bucket_size)
                    seq_emb = tf.nn.embedding_lookup(embeddings, indices, name="deep_shared_lookup_" + col + str(shared_flag))
                    if col.endswith('seq'):
                        print("into...")
                        seq_emb = tf.reduce_mean(seq_emb,1)
                    deep_sum.append(seq_emb)
            for col in CONTINUOUS_COLS:
                deep_sum.append(tf.expand_dims(tf.to_float(features[col]), -1, name='continuous_' + col))
            curr_layer = tf.concat(deep_sum, 1, name="deep_inputs_layer")

            # Build the DNN

            for index, layer_size in enumerate(_HIDDEN_UNITS):
                curr_layer = tf.layers.dense(
                    curr_layer,
                    layer_size,
                    activation=tf.nn.relu,
                    kernel_initializer=init_ops.glorot_uniform_initializer(),
                    name="deep_hidden_layer" + str(index)
                )
            deep_logits = tf.layers.dense(curr_layer, units=1, name="deep_logits")
    ####################################
    return deep_logits

def _MY_HEAD(mode,
             label_ctr,
             label_cvr,
             ctcvr_prob,
             ctr_prob,
             cvr_prob,
             ctr_logits,
             cvr_logits,
             dnn_learning_rate):

    _DNN_LEARNING_RATE =dnn_learning_rate
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': ctcvr_prob,
            'ctr_probabilities': ctr_prob,
            'cvr_probabilities': cvr_prob
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ctr,logits=ctr_logits),name='ctr_loss')
    cvr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cvr,logits=cvr_logits),name='cvr_loss')
    loss = tf.add(ctr_loss,cvr_loss,name='ctcvr_loss')
    ctr_accuracy = tf.metrics.accuracy(labels=label_ctr,
                                       predictions=tf.to_float(tf.greater_equal(ctr_prob, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(labels=label_cvr, predictions=tf.to_float(tf.greater_equal(cvr_prob, 0.5)))
    ctr_auc = tf.metrics.auc(label_ctr, ctr_prob)
    cvr_auc = tf.metrics.auc(label_cvr, cvr_prob)
    metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=_DNN_LEARNING_RATE)
    train_op = dnn_optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def esmm(features,
         labels,
         params,
         mode):

    dnn_learning_rate = params['_DNN_LEARNING_RATE']
    label_ctr = tf.reshape(labels['click'],(-1,1))
    label_cvr = tf.reshape(labels['buy'],(-1,1))

    with variable_scope.variable_scope(
            'CTR_Module',
            values=tuple(six.itervalues(features)),
    ) as scope:
        ctr_logits = dnn_model(features,params)
        ctr_prob = tf.sigmoid(ctr_logits)
    with variable_scope.variable_scope(
            'CVR_Module',
            values=tuple(six.itervalues(features)),
    ) as scope:
        cvr_logits = dnn_model(features,params)
        cvr_prob = tf.sigmoid(cvr_logits)

    ctcvr_prob = tf.multiply(ctr_prob,cvr_prob,name='ctrcvr_prob')

    return _MY_HEAD(
        mode,
        label_ctr,
        label_cvr,
        ctcvr_prob,
        ctr_prob,
        cvr_prob,
        ctr_logits,
        cvr_logits,
        dnn_learning_rate)



