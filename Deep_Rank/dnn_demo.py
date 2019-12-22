#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:05 PM
'''
import  tensorflow as tf
from model_brain import  BaseModel
'''DNN demo'''
class DNN(BaseModel):
    '''DNN model 最简单的dnn结构'''
    def __init__(self, features, labels, params, mode):
        super(DNN,self).__init__(features, labels, params, mode)
        _,self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('DNN_Module'):
            self.logits = self._model_fn

    @property
    def _model_fn(self):
        '''DNN model'''
        with tf.variable_scope('fc_net'):
            logits = self.fc_net(self.embedding_layer,1)
        return logits