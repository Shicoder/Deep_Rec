#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : shixiangfu
import sys
import tensorflow as tf
from tensorflow.python.estimator.canned import head
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.training import training_util
from tensorflow.python.estimator.canned import optimizers
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
import six
###############
def wide_and_deep(features=None,params=None):
    ###############
    WIDE_CATE_COLS = params['WIDE_CATE_COLS']
    CONTINUOUS_COLS = params['CONTINUOUS_COLS']
    DEEP_EMBEDDING_COLS = params['DEEP_EMBEDDING_COLS']
    WIDE_CROSS_COLS = params['WIDE_CROSS_COLS']
    DEEP_SHARED_EMBEDDING_COLS = params['DEEP_SHARED_EMBEDDING_COLS']
    _HIDDEN_UNITS = params['_HIDDEN_UNITS']
    _LINEAR_LEARNING_RATE = params['_LINEAR_LEARNING_RATE']
    _DNN_LEARNING_RATE = params['_DNN_LEARNING_RATE']


    wide_logits = None
    linear_absolute_scope = None
    if params['WIDE']:
        wide_sum = []
        with variable_scope.variable_scope(
                'linear',
                values=tuple(six.itervalues(features))) as scope:
            linear_absolute_scope = scope.name
            for col, size in WIDE_CATE_COLS:
                w_wide = tf.get_variable(shape=[size, 1], initializer=init_ops.zeros_initializer, trainable=True,
                                         name="Wide_Part_Weights_Cate" + col)
                indices = string_ops.string_to_hash_bucket_fast(
                    features[col], size, name="wide_hash_" + col)
                wide_sum.append(tf.nn.embedding_lookup(w_wide, indices, name="wide_cat_lookup_" + col))
            # for col, size in WIDE_BUCKET_COLS:
            #     w_wide = tf.get_variable(shape=[size, 1], initializer=init_ops.zeros_initializer, trainable=True,
            #                              name="Wide_Part_Weights_Bucket" + col)
            #     indices = string_ops.string_to_hash_bucket_fast(
            #         features[col], size, name="wide_hash_" + col)
            #     wide_sum.append(tf.nn.embedding_lookup(w_wide, indices, name="wide_bucket_lookup_" + col))
            for col1, col2, size in WIDE_CROSS_COLS:
                w_wide = tf.get_variable(shape=[size, 1], initializer=init_ops.zeros_initializer, trainable=True,
                                         name="Wide_Part_Weights_Cross" + col1 + '_' + col2)
                # cross_input = tf.as_string(tf.string_to_number(features[col1],_dtypes.int64)*tf.string_to_number(features[col2],_dtypes.int64))
                cross_input = tf.string_join([features[col1], features[col2]], separator="_")
                indices = string_ops.string_to_hash_bucket_fast(
                    cross_input, size, name="wide_hash_" + col1 + '_' + col2)
                wide_sum.append(tf.nn.embedding_lookup(w_wide, indices, name="wide_cross_lookup_" + col1 + '_' + col2))

            w_wide = tf.get_variable(shape=[len(CONTINUOUS_COLS), 1], initializer=init_ops.zeros_initializer,
                                     trainable=True,
                                     name="Wide_Part_Weights_Continus")
            bias = tf.get_variable(shape=[1], initializer=init_ops.zeros_initializer, trainable=True,
                                   name="Wide_Part_Bias")
            x = tf.concat([tf.expand_dims(tf.to_float(features[col]), -1) for col in CONTINUOUS_COLS], 1,
                          name='continus_concat')
            continue_logits = tf.matmul(x, w_wide) + bias

            wide_logits = tf.reduce_sum(wide_sum, 0)
            wide_logits += continue_logits
    ##################
    deep_logits = None
    dnn_absolute_scope = None
    if params['DEEP']:
        # with tf.variable_scope('Deep_model'):
        with variable_scope.variable_scope(
                'Deep_model',
                values=tuple(six.itervalues(features)),
        ) as scope:
            dnn_absolute_scope = scope.name
            # Convert categorical (string) values to embeddings
            deep_sum = []
            for col, vals, embedding_size, col_type in DEEP_EMBEDDING_COLS:
                bucket_size = vals if isinstance(vals, int) else len(vals)
                # embed_initializer = tf.truncated_normal_initializer(
                #     stddev=(1.0 / tf.sqrt(float(embedding_size))))
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
                if col_type == 'seq':
                    print("test my seq:",col)
                    seq_emb = tf.reduce_mean(seq_emb,1)
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

    my_head = head._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
        loss_reduction=losses.Reduction.SUM)
    print(my_head.logits_dimension)

    if deep_logits is not None and wide_logits is not None:
        logits = deep_logits + wide_logits
    elif deep_logits is not None:
        logits = deep_logits
    else:
        logits = wide_logits

    dnn_optimizer = optimizers.get_optimizer_instance(
        'Adagrad', learning_rate=_DNN_LEARNING_RATE)

    def _linear_learning_rate(num_linear_feature_columns):
        default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
        return min(_LINEAR_LEARNING_RATE, default_learning_rate)

    linear_optimizer = optimizers.get_optimizer_instance(
        'Ftrl',
        learning_rate=_linear_learning_rate(len(WIDE_CATE_COLS)))

    def _train_op_fn(loss):
        train_ops = []
        global_step = training_util.get_global_step()
        if deep_logits is not None:
            train_ops.append(
                dnn_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=dnn_absolute_scope)))
        if wide_logits is not None:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=linear_absolute_scope)))

        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            return state_ops.assign_add(global_step, 1).op
    return my_head,logits,_train_op_fn