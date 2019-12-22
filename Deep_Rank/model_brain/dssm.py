#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:21 PM
'''
import tensorflow as tf
from model_brain import BaseModel
class DSSM(BaseModel):
    '''DSSM 本来是用于生成语言向量的，不过双塔结构和推荐，广告等场景中的user/item类似，
    因此也会被用于计算user embedding 和item embedding之间的相关度'''
    def __init__(self, features, labels, params, mode):
        super(DSSM,self).__init__(features, labels, params, mode)
        self.user_Features, self.item_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            '''liner侧放user feature，deep侧放item feature'''
            self.user_embeddings_layer = self.get_input_layer(self.user_Features)
            self.item_embeddings_layer = self.get_input_layer(self.item_Features)
        with tf.variable_scope('DSSM_Module'):
            self.logits = self._model_fn
    @property
    def _model_fn(self):
        '''DNN model'''
        with tf.variable_scope('user_embedding'):
            user_embedding = self.fc_net(self.user_embeddings_layer,8,activation='prelu')
        with tf.variable_scope('item_embedding'):
            item_embedding = self.fc_net(self.item_embeddings_layer, 8,activation='prelu')
        user_norm = tf.sqrt(tf.reduce_sum(tf.square(user_embedding),1))
        print("user:",user_norm.get_shape().as_list())
        print("user_embedding:", user_embedding.get_shape().as_list())
        item_norm = tf.sqrt(tf.reduce_sum(tf.square(item_embedding),1))
        prod = tf.reduce_sum(tf.multiply(user_embedding,item_embedding),1)
        print("prod",prod.get_shape().as_list())
        logits = tf.div(prod, user_norm * item_norm + 1e-8, name="scores")
        logits = tf.expand_dims(logits, axis=-1)
        return logits