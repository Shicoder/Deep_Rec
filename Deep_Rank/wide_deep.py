#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:11 PM
'''
import  tensorflow as tf
from model_brain import BaseModel
import six
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.estimator.canned import head
from tensorflow.python.ops.losses import losses

class WD_Model(BaseModel):
    '''wide and deep model'''
    def __init__(self, features, labels, params, mode):
        super(WD_Model,self).__init__(features, labels, params, mode)
        self.Linear_Features,self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('DNN_Module'):
            self.logits,self.train_op_fn = self._model_fn


    @property
    def _model_fn(self):
        '''wide and deep model'''
        with tf.variable_scope('fc_net'):
            with tf.variable_scope(
                    'deep_model',
                    values=tuple(six.itervalues(self.features)),
            ) as scope:
                dnn_absolute_scope = scope.name
                dnn_logits = self.fc_net(self.embedding_layer,1)
            with tf.variable_scope(
                    'linear_model',
                    values=tuple(six.itervalues(self.features)),
            ) as scope:
                linear_absolute_scope = scope.name
                linear_logits = tf.feature_column.linear_model(self.features,self.Linear_Features)

            if dnn_logits is not None and linear_logits is not None:
                logits = dnn_logits + linear_logits
            elif dnn_logits is not None:
                logits = dnn_logits
            else:
                logits = linear_logits

            dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])

            def _linear_learning_rate(num_linear_feature_columns):
                default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
                return min(self.params['LINEAR_LEARNING_RATE'], default_learning_rate)

            linear_optimizer = tf.train.FtrlOptimizer(_linear_learning_rate(len(self.Linear_Features)))
            def _train_op_fn(loss):
                train_ops = []
                global_step = tf.train.get_global_step()
                if dnn_logits is not None:
                    train_ops.append(
                        dnn_optimizer.minimize(
                            loss,
                            var_list=ops.get_collection(
                                ops.GraphKeys.TRAINABLE_VARIABLES,
                                scope=dnn_absolute_scope)))
                if linear_logits is not None:
                    train_ops.append(
                        linear_optimizer.minimize(
                            loss,
                            var_list=ops.get_collection(
                                ops.GraphKeys.TRAINABLE_VARIABLES,
                                scope=linear_absolute_scope)))

                train_op = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op]):
                    return state_ops.assign_add(global_step, 1).op
            return logits, _train_op_fn

    @property
    def build_estimator_spec(self):
        '''Build EstimatorSpec'''
        my_head = head._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                loss_reduction=losses.Reduction.SUM)
        return my_head.create_estimator_spec(
            features=self.features,
            mode=self.mode,
            labels=self.labels,
            train_op_fn=self.train_op_fn,
            logits=self.logits)