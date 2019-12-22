#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:19 PM
'''
import tensorflow as tf
from model_brain import BaseModel

class DeepFM(BaseModel):
    '''
    DeepFM model
    model feature 中
    wide配置直接使用正常wide and deep 方式配置
    deep配置只能配置embeddingcolumn，并且所有的column的dimension必须一致；
    '''
    def __init__(self, features, labels, params, mode):
        super(DeepFM,self).__init__(features, labels, params, mode)
        self.Linear_Features, self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('DeepFM_Module'):
            self.logits = self._model_fn

    @property
    def _model_fn(self):
        linear_logits = tf.feature_column.linear_model(self.features,self.Linear_Features)
        mlp_logits = self.fc_net(self.embedding_layer,1)
        fm_logits = self.fm_layer(self.embedding_layer)
        fm_logits = tf.expand_dims(fm_logits,-1)
        print(fm_logits.get_shape().as_list())
        last_layer = tf.concat([linear_logits,mlp_logits, fm_logits], 1)
        logits = tf.layers.dense(last_layer,1)
        return logits

    def fm_layer(self,embedding_layer):
        """"""
        # embedding_layer shape is [batch_size, column_num*embedding_size]
        # shape = [batch_size, column_num, embedding_size]
        dimension = self._check_columns_dimension(self.Deep_Features)
        column_num = len(self.Deep_Features)

        # dimension = self.Deep_Features[0].dimension
        print("column_num:",column_num)
        print("dimension:",dimension)
        net = tf.reshape(embedding_layer, (-1, column_num, dimension), "fm_inputs")

        # sum-square-part
        fm_sum_square = tf.square(tf.reduce_sum(net, 1))

        # squre-sum-part...2
        fm_squared_sum = tf.reduce_sum(tf.square(net),1)

        # second order...3
        logits = tf.reduce_sum(0.5 * tf.subtract(fm_sum_square, fm_squared_sum), -1)
        return logits