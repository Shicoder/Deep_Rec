#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:07 PM
'''
import  tensorflow as tf
from model_brain import  BaseModel
class DCN(BaseModel):
    '''Deep cross network'''
    def __init__(self, features, labels, params, mode):
        super(DCN,self).__init__(features, labels, params, mode)
        self.cross_layer_num = params["CROSS_LAYER_NUM"]
        _,self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('DCN_Module'):
            self.logits = self._model_fn

    @property
    def _model_fn(self):
        '''dcn model'''
        mlp_layer = self.fc_net(self.embedding_layer,8,'relu')
        cross_layer = self.cross_net(self.embedding_layer,self.cross_layer_num)
        last_layer = tf.concat([mlp_layer, cross_layer], 1)
        logits = tf.layers.dense(last_layer,1)
        return logits