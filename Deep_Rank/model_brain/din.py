#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:14 PM
'''
import tensorflow as tf
from Deep_Rank.model_brain import BaseModel

class DIN(BaseModel):
    '''Deep Interest Network Model'''
    def __init__(self, features, labels, params, mode):
        super(DIN,self).__init__(features, labels, params, mode)
        # seq feature,
        self.din_user_goods_seq = features["seq_goods_id_seq"] # can not include in model feature
        self.din_target_goods_id = features["goods_id"] # can not include in model feature or new type
        self.goods_embedding_size = 16
        self.goods_bucket_size = 1000
        self.goods_attention_hidden_units = [50, 25]

        # self.din_user_class_seq = features["class_seq"]
        # self.din_target_class_id = features["class_id"]
        _, self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.common_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('Din_Module'):
            self.logits = self._model_fn


    @property
    def _model_fn(self):
        # feature_columns not include attention feature
        din_user_seq = tf.string_to_hash_bucket_fast(self.din_user_goods_seq,self.goods_bucket_size)
        din_target_id = tf.string_to_hash_bucket_fast(self.din_target_goods_id,self.goods_bucket_size)
        din_useq_embedding, din_tid_embedding = self.attention_layer(din_user_seq, din_target_id,
                                                                     self.goods_bucket_size,
                                                                     self.goods_embedding_size,
                                                                     self.goods_attention_hidden_units,
                                                                     id_type="click_seq")
        din_net = tf.concat([self.common_layer, din_useq_embedding, din_tid_embedding], axis=1)
        logits = self.fc_net(din_net,1)
        return logits