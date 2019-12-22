#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:20 PM
'''
import tensorflow as tf
from model_brain import BaseModel
class xDeepFM(BaseModel):
    '''
    DeepFM model
    model feature 中
    wide配置直接使用正常wide and deep 方式配置
    deep配置只能配置embeddingcolumn，并且所有的column的dimension必须一致'''
    def __init__(self, features, labels, params, mode):
        super(xDeepFM,self).__init__(features, labels, params, mode)
        self.field_nums = [10,10]
        self.Linear_Features, self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('xDeepFM_Module'):
            self.logits = self._model_fn

    @property
    def _model_fn(self):
        linear_layer = tf.feature_column.linear_model(self.features,self.Linear_Features)
        cin_layer = self.cin_net(self.embedding_layer,direct=False, residual=True)
        dnn_layer = self.fc_net(self.embedding_layer,1)

        linear_logit = tf.layers.dense(linear_layer,1)
        cin_logit = tf.layers.dense(cin_layer,1)
        dnn_logit = tf.layers.dense(dnn_layer,1)

        last_layer = tf.concat([linear_logit, cin_logit, dnn_logit], 1)
        logits = tf.layers.dense(last_layer,1)
        return logits