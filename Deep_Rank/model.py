#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
# @Author  : shixiangfu
import tensorflow as tf
import sys
sys.path.append("..")
from tensorflow.python.ops import string_ops,array_ops,math_ops
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.estimator.export import export_output
from transform_feature import FeatureBuilder
from rnn import dynamic_rnn
from alg_utils.utils_tf import VecAttGRUCell

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops import init_ops
from tensorflow.python.estimator.canned import head
from tensorflow.python.ops.losses import losses
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
import six
import math
from alg_utils.utils_tf import PReLU,get_input_schema_spec,dice
import random

class BaseModel(object):
    def __init__(self, features,labels,params,mode):
        self.features = features
        self.labels =labels
        self.params = params
        self.mode = mode
        self.model_features = params["FEATURES_DICT"]
        self.logits = None
        self.Wide_Features,\
        self.Deep_Features = self._get_feature_embedding
        self.embedding_layer = None

    @property
    def _get_feature_embedding(self):
        Feature_Columns = FeatureBuilder(self.model_features)
        WideFeats,DeepFeas = Feature_Columns.get_feature_columns()
        return WideFeats,DeepFeas

    def get_embedding_layer(self,feature_columns):
        embedding_layer = tf.feature_column.input_layer(self.features, feature_columns)
        return embedding_layer

    def _classification_output(self,scores, n_classes, label_vocabulary=None):
        batch_size = array_ops.shape(scores)[0]
        if label_vocabulary:
            export_class_list = label_vocabulary
        else:
            export_class_list = string_ops.as_string(math_ops.range(n_classes))
        export_output_classes = array_ops.tile(
            input=array_ops.expand_dims(input=export_class_list, axis=0),
            multiples=[batch_size, 1])
        return export_output.ClassificationOutput(
            scores=scores,
            # `ClassificationOutput` requires string classes.
            classes=export_output_classes)

    def embedding_table(self,bucket_size,embedding_size,col):
        embeddings = tf.get_variable(
            shape=[bucket_size, embedding_size],
            initializer=init_ops.glorot_uniform_initializer(),
            dtype=tf.float32,
            name="deep_embedding_" + col)
        return embeddings
    def get_activation(self,activation):
        if activation==None:
            act = None
        elif activation == 'prelu':
            act = PReLU
        elif activation == 'relu':
            act = tf.nn.relu
        else:
            act = tf.nn.sigmoid
        return act

    # get_optimizer_instance
    def fc_net(self,net,last_num=1,activation=None):
        '''MLP'''

        act  = self.get_activation(activation)
        net = tf.layers.batch_normalization(inputs=net, name='bn1', training=True)
        for units in self.params['HIDDEN_UNITS']:
            net = tf.layers.dense(net, units=units, activation=PReLU)
            if 'DROPOUT_RATE' in self.params and self.params['DROPOUT_RATE'] > 0.0:
                net = tf.layers.dropout(net, self.params['DROPOUT_RATE'], training=(self.mode == tf.estimator.ModeKeys.TRAIN))
        logits = tf.layers.dense(net, last_num, activation=act)
        return logits

    def cross_layer(self,x0,x,name):
        with tf.variable_scope(name):
            input_dim = x0.get_shape().as_list()[1]
            w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
            xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
            return x0 * xb + b + x

    def cross_net(self,net,cross_layer_num):
        '''cross network'''
        x0 = net
        for i in range(cross_layer_num):
            net = self.cross_layer(x0,net,'cross_{}'.format(i))
        return net


    def build_estimator_spec(self):
        '''Build EstimatorSpec'''
        my_head = head._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                loss_reduction=losses.Reduction.SUM)
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        return my_head.create_estimator_spec(
            features=self.features,
            mode=self.mode,
            labels=self.labels,
            optimizer=optimizer,
            logits=self.logits)

    # tf.estimator.DNNLinearCombinedClassifier（）

    def attention_layer(self, seq_ids, tid, bucket_size, embedding_size, attention_hidden_units, id_type):
        with tf.variable_scope("attention_" + id_type):
            embeddings = self.embedding_table(bucket_size, embedding_size, id_type)
            seq_emb = tf.nn.embedding_lookup(embeddings,
                                                 seq_ids)  # shape(batch_size, max_seq_len, embedding_size)
            u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])

            tid_emb = tf.nn.embedding_lookup(embeddings, tid)  # shape(batch_size, embedding_size)
            max_seq_len = tf.shape(seq_ids)[1]  # padded_dim
            a_emb = tf.reshape(tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, embedding_size])

            net = tf.concat([u_emb, a_emb, u_emb - a_emb, u_emb * a_emb], axis=1)
            for units in attention_hidden_units:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
            att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")
            wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
            # masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
            masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)
            att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
            return att_emb, tid_emb

'''DNN demo'''
class DNN(BaseModel):
    '''DNN model'''
    def __init__(self, features, labels, params, mode):
        super(DNN,self).__init__(features, labels, params, mode)
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_embedding_layer(self.Deep_Features)
        with tf.variable_scope('DNN_Module'):
            self.logits = self._model_fn

    @property
    def _model_fn(self):
        '''DNN model'''
        with tf.variable_scope('fc_net'):
            logits = self.fc_net(self.embedding_layer,1)
        return logits

class DCN(BaseModel):
    '''Deep cross network'''
    def __init__(self, features, labels, params, mode):
        super(DCN,self).__init__(features, labels, params, mode)
        self.cross_layer_num = params["CROSS_LAYER_NUM"]
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_embedding_layer(self.Deep_Features)
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

class WD_Model(BaseModel):
    '''wide and deep model'''
    def __init__(self, features, labels, params, mode):
        super(WD_Model,self).__init__(features, labels, params, mode)
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_embedding_layer(self.Deep_Features)
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
                linear_logits = tf.feature_column.linear_model(self.features,self.Wide_Features)

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

            linear_optimizer = tf.train.FtrlOptimizer(_linear_learning_rate(len(self.Wide_Features)))
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
        with tf.variable_scope('Embedding_Module'):
            self.common_layer = self.get_embedding_layer(self.Deep_Features)
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

class DeepFM(BaseModel):
    def __init__(self, features, labels, params, mode):
        super(DeepFM,self).__init__(features, labels, params, mode)
        self.cross_layer_num = params["CROSS_LAYER_NUM"]
        with tf.variable_scope('DCN_module'):
            self.logits = self.deepfm()
    def deepfm(self):
        embedding_layer = tf.feature_column.input_layer(self.features, self.Deep_Features)
        mlp_layer = self.fc_net(embedding_layer,8,'relu')
        fm_layer = self.fm_layer(embedding_layer)
        logits = tf.concat([mlp_layer, fm_layer], 1)
        return logits

    def fm_layer(self,features):
        """FM model .
        """
        # flat_val = tf.feature_column.input_layer(self.features, self.Deep_Features)\
        # # shape(batch_size, column_num * embedding_size)
        # vals = tf.reshape(flat_val, (-1, column_num, dimension), "interaction_embeddings")
        # # sum-square-part
        # summed_val = math_ops.reduce_sum(vals, 1)
        # summed_square_val = math_ops.square(summed_val)
        #
        # # squre-sum-part...2
        # squared_val = math_ops.square(vals)
        # squared_sum_val = math_ops.reduce_sum(squared_val, 1)
        #
        # # second order...3
        # logits = math_ops.reduce_sum(0.5 * math_ops.subtract(summed_square_val, squared_sum_val), -1)
        # return logits
        pass




class DIEN(BaseModel):
    '''Deep Interest Evolution Network Model'''
    def __init__(self, features, labels, params, mode):
        super(DIEN,self).__init__(features, labels, params, mode)
        # seq feature,
        self.din_user_goods_seq = features["seq_goods_id_seq"]
        self.din_target_goods_id = features["goods_id"]
        self.goods_embedding_size = 16
        self.goods_bucket_size = 1000
        self.goods_attention_hidden_units = [50, 25]

        self.uid = None

        self.GRU_HIDDEN_SIZE = 8
        self.ATTENTION_SIZE = 8
        self.sequence_length = 6

        # self.din_user_class_seq = features["class_seq"]
        # self.din_target_class_id = features["class_id"]
        self.logits = self.dien_model()

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)

        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def din_fcn_attention(self,query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                          return_alphas=False, forCnn=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 1)

        # if time_major:
        #     # (T,B,D) => (B,T,D)
        #     facts = tf.array_ops.transpose(facts, [1, 0, 2])
        # Trainable parameters
        mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
        querry_size = query.get_shape().as_list()[-1]
        query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
        query = PReLU(query)
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if return_alphas:
            return output, scores
        return output

    def dien_model(self):
        # history embedding
        common_layer = tf.feature_column.input_layer(self.features, self.Deep_Features)
        din_user_seq = tf.string_to_hash_bucket_fast(self.din_user_goods_seq,self.goods_bucket_size)
        din_target_id = tf.string_to_hash_bucket_fast(self.din_target_goods_id,self.goods_bucket_size)
        name = "goods"
        embedding_size = 8
        bucket_size = 1000
        with tf.variable_scope("seq_embedding_table"+name):
            embeddings = self.embedding_table(bucket_size, embedding_size, name)
            seq_emb = tf.nn.embedding_lookup(embeddings,
                                             din_user_seq)  # shape(batch_size, max_seq_len, embedding_size)
            u_his_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])

            tid_emb = tf.nn.embedding_lookup(embeddings, din_target_id)


        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.GRU_HIDDEN_SIZE), inputs=u_his_emb,
                                         sequence_length=self.sequence_length,
                                         dtype=tf.float32,
                                         scope="gru1")
        # from tensorflow.contrib import legacy_seq2seq
        # legacy_seq2seq.sequence_loss_by_example()
        # tf.diag_part(yhat)
        # tf.matmul(output, sampled_W, transpose_b=True) 利用转至矩阵来做；

        x = tf.shape(u_his_emb)[0]
        y = tf.shape(u_his_emb)[1]
        tmp_his_emb = tf.reshape(u_his_emb,[-1,1])
        tf.random.shuffle(tmp_his_emb)
        noclk_u_his_emb = tf.reshape(tmp_his_emb,[x,y])

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], u_his_emb[:, 1:, :],
                                         noclk_u_his_emb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = self.din_fcn_attention(tid_emb, rnn_outputs, self.ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.GRU_HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)
        # self.uid
        item_his_eb_sum = tf.reduce_sum(u_his_emb,1)
        inp = tf.concat([common_layer, tid_emb, item_his_eb_sum, tid_emb * item_his_eb_sum, final_state2], 1)
        logits = self.fc_net(inp,1)
        return logits

    def build_estimator_spec(self):
        '''Build EstimatorSpec'''

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            CLASSES = 'classes'
            CLASS_IDS = 'class_ids'
            two_class_logits = tf.concat(
                (array_ops.zeros_like(self.logits), self.logits),
                axis=-1, name='two_class_logits')
            probabilities = tf.nn.softmax(
                two_class_logits, name='probabilities')
            class_ids = tf.argmax(two_class_logits, axis=-1, name=CLASS_IDS)
            class_ids = tf.expand_dims(class_ids, axis=-1)
            classes = tf.as_string(class_ids, name='str_classes')
            classifier_output = self._classification_output(
                scores=probabilities, n_classes=2,
                label_vocabulary=None)
            predictions = {
                'probabilities': probabilities,
                CLASS_IDS: class_ids,
                CLASSES: classes
            }
            _DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            _CLASSIFY_SERVING_KEY = 'classification'
            _REGRESS_SERVING_KEY = 'regression'
            _PREDICT_SERVING_KEY = 'predict'
            export_outputs = {
                _DEFAULT_SERVING_KEY: classifier_output,
                _CLASSIFY_SERVING_KEY: classifier_output,
                _PREDICT_SERVING_KEY: tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(self.mode, predictions=predictions, export_outputs=export_outputs)

        click_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.reshape(self.labels['ctr'],(-1,1)),
            logits=self.logits),
            name="ctr_loss")

        lambda_a = 0.3
        loss = tf.add(click_loss, lambda_a*self.aux_loss, name="total_loss")

        accuracy = tf.metrics.accuracy(labels=self.labels['ctr'],
                                           predictions=tf.to_float(tf.greater_equal(tf.sigmoid(self.logits), 0.5)))
        auc = tf.metrics.auc(self.labels['ctr'], tf.sigmoid(self.logits))

        metrics = {'accuracy': accuracy, 'auc': auc}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)

        # Create training op.
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)


class ESMM(BaseModel):
    def __init__(self, features, labels, params, mode):
        super(ESMM,self).__init__(features, labels, params, mode)
        # seq feature,
        self.din_user_goods_seq = features["seq_goods_id_seq"]
        self.din_target_goods_id = features["goods_id"]
        self.goods_embedding_size = 16
        self.goods_bucket_size = 1000
        self.goods_attention_hidden_units = [50, 25]

        self.uid = None

        self.GRU_HIDDEN_SIZE = 8
        self.ATTENTION_SIZE = 8
        self.sequence_length = 6

        # self.din_user_class_seq = features["class_seq"]
        # self.din_target_class_id = features["class_id"]


        self.logits = self.esmm()

    def esmm(self):
        with tf.variable_scope('embedding_module'):
            embedding_layer = tf.feature_column.input_layer(self.features, self.Deep_Features)
        with tf.variable_scope('ctr_model'):
            ctr_logits = self.fc_net(embedding_layer)
            # ctr_logits = DIN(self.features, self.labels, self.params, self.mode)
        with tf.variable_scope('cvr_model'):
            cvr_logits = self.fc_net(embedding_layer)
            # cvr_logits = DIN(self.features, self.labels, self.params, self.mode)
        ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
        cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")

    def build_estimator_spec(self):
        pass

    pass

class BST(BaseModel):
    pass

class PNN(BaseModel):
    pass

class DSSM(BaseModel):
    pass

class IRGAN(BaseModel):
    pass

class DSIN(BaseModel):
    pass

# class dien(object):
#     def __init__(self, features,labels,params,mode):
#         self.features = features
#         self.labels =labels
#         self.params = params
#         self.mode = mode
#         self.model_features = params["FEATURES_DICT"]
#
#         self.din_user_goods_seq = features["seq_goods_id_seq"]
#         self.din_target_goods_id = features["goods_id"]
#         self.din_user_class_seq = features["class_seq"]
#         self.din_target_class_id = features["class_id"]
#
#     def get_feature_columns(self):
#         Feature_Columns = FeatureBuilder(self.model_features)
#         _,DeepFeatures = Feature_Columns.get_feature_columns()
#         return DeepFeatures
#
#     def _classification_output(self,scores, n_classes, label_vocabulary=None):
#         batch_size = array_ops.shape(scores)[0]
#         if label_vocabulary:
#             export_class_list = label_vocabulary
#         else:
#             export_class_list = string_ops.as_string(math_ops.range(n_classes))
#         export_output_classes = array_ops.tile(
#             input=array_ops.expand_dims(input=export_class_list, axis=0),
#             multiples=[batch_size, 1])
#         return export_output.ClassificationOutput(
#             scores=scores,
#             # `ClassificationOutput` requires string classes.
#             classes=export_output_classes)
#
#     def embedding_table(self,bucket_size,embedding_size,col):
#         embeddings = tf.get_variable(
#             shape=[bucket_size, embedding_size],
#             initializer=init_ops.glorot_uniform_initializer(),
#             dtype=tf.float32,
#             name="deep_embedding_" + col)
#         return embeddings
#
#     def fc_net(self,net):
#         '''MLP'''
#         net = tf.layers.batch_normalization(inputs=net, name='bn1', training=True)
#         for units in self.params['HIDDEN_UNITS']:
#             net = tf.layers.dense(net, units=units, activation=PReLU)
#             if 'DROPOUT_RATE' in self.params and self.params['DROPOUT_RATE'] > 0.0:
#                 net = tf.layers.dropout(net, self.params['DROPOUT_RATE'], training=(self.mode == tf.estimator.ModeKeys.TRAIN))
#         logits = tf.layers.dense(net, 1, activation=None)
#         return logits
#
#
#     def Din_model(self,feature_columns):
#         # feature_columns not include attention feature
#         common_layer = tf.feature_column.input_layer(self.features, feature_columns)
#         din_user_seq = tf.string_to_hash_bucket_fast(self.din_user_goods_seq)
#         din_target_id = tf.string_to_hash_bucket_fast(self.din_target_goods_id)
#         din_useq_embedding,din_tid_embedding = self.attention_layer(din_user_seq,din_target_id,id_type="click_seq")
#         din_net = tf.concat([common_layer,din_useq_embedding,din_tid_embedding],axis=1)
#         logits = self.fc_net(din_net)
#         return logits
#
#     def attention_layer(self, seq_ids, tid, id_type):
#         with tf.variable_scope("attention_" + id_type):
#             embedding_size = 16
#             bucket_size = 10000000
#             embeddings = self.embedding_table(bucket_size,embedding_size,id_type)
#             seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids)  # shape(batch_size, max_seq_len, embedding_size)
#             u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])
#
#             tid_emb = tf.nn.embedding_lookup(embeddings, tid)  # shape(batch_size, embedding_size)
#             max_seq_len = tf.shape(seq_ids)[1]  # padded_dim
#             a_emb = tf.reshape(tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, embedding_size])
#
#             net = tf.concat([u_emb, a_emb,u_emb - a_emb,u_emb * a_emb], axis=1)
#             attention_hidden_units = [50,25]
#             for units in attention_hidden_units:
#                 net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
#             att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
#             att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")
#             wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
#             # masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
#             masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)
#             att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
#             return att_emb, tid_emb
#
#     def dien_layer(self,seq_ids,tid,id_type):
#         with tf.variable_scope("RNN_layer_" + id_type):
#             embedding_size = 16
#             bucket_size = 10000000
#             sequence_length = 20
#             embeddings = self.embedding_table(bucket_size, embedding_size, id_type)
#             seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids)  # shape(batch_size, max_seq_len, embedding_size)
#             u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])
#
#             rnn_outputs, _ = dynamic_rnn(GRUCell(embedding_size), inputs=u_emb,
#                                          sequence_length=sequence_length, dtype=tf.float32,
#                                          scope="GRU1")
#             tf.summary.histogram('GRU_outputs', rnn_outputs)
#
#
#             return
#
#     def Dien_model(self,feature_columns):
#         common_layer = tf.feature_column.input_layer(self.features, feature_columns)
#         din_user_seq = tf.string_to_hash_bucket_fast(self.din_user_goods_seq)
#         din_target_id = tf.string_to_hash_bucket_fast(self.din_target_goods_id)
#         din_useq_embedding,din_tid_embedding = self.dien_layer(din_user_seq,din_target_id,id_type="click_seq")
#         din_net = tf.concat([common_layer,din_useq_embedding,din_tid_embedding],axis=1)
#         logits = self.fc_net(din_net)
#         return logits
#
#
#     def dcn_model(self):
#         '''dcn model '''
#         pass
#
#     def Dnn_Model(self,feature_columns):
#         '''dnn model'''
#         net = tf.feature_column.input_layer(self.features, feature_columns)
#         # Build the hidden layers, sized according to the 'hidden_units' param.
#         logits = self.fc_net(net)
#         return logits

    # def Build_EstimatorSpec(self):
    #     '''Build EstimatorSpec'''
    #     with tf.variable_scope('embedding_module'):
    #         feature_columns = self.get_feature_columns()
    #         print("feature_columns:", feature_columns)
    #     with tf.variable_scope('ctr_model'):
    #         ctr_logits = self.Dien_model(feature_columns)
    #
    #     ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    #     cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    #     prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    #     if self.mode == tf.estimator.ModeKeys.PREDICT:
    #         CLASSES = 'classes'
    #         CLASS_IDS = 'class_ids'
    #         two_class_ctcvr_prob = tf.concat(
    #             (tf.subtract(1.0, prop), prop),
    #             # (array_ops.zeros_like(ctcvr_prob), ctcvr_prob),
    #             axis=-1, name='two_class_logits')
    #         class_ids = tf.argmax(two_class_ctcvr_prob, axis=-1, name=CLASS_IDS)
    #         class_ids = tf.expand_dims(class_ids, axis=-1)
    #
    #         classes = tf.as_string(class_ids, name='str_classes')
    #         classifier_output = self._classification_output(
    #             scores=two_class_ctcvr_prob, n_classes=2,
    #             label_vocabulary=None)
    #         predictions = {
    #             'probabilities': prop,
    #             CLASS_IDS: class_ids,
    #             CLASSES: classes,
    #             'ctr_probabilities': ctr_predictions,
    #             'cvr_probabilities': cvr_predictions
    #         }
    #         _DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    #         _CLASSIFY_SERVING_KEY = 'classification'
    #         _REGRESS_SERVING_KEY = 'regression'
    #         _PREDICT_SERVING_KEY = 'predict'
    #         export_outputs = {
    #             _DEFAULT_SERVING_KEY: classifier_output,
    #             _CLASSIFY_SERVING_KEY: classifier_output,
    #             _PREDICT_SERVING_KEY: tf.estimator.export.PredictOutput(predictions)
    #         }
    #         return tf.estimator.EstimatorSpec(self.mode, predictions=predictions, export_outputs=export_outputs)
    #
    #     y = self.labels['cvr']
    #     cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(tf.reshape(y,(-1,1)), prop), name="cvr_loss")
    #     ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.labels['ctr'],(-1,1)), logits=ctr_logits),
    #                              name="ctr_loss")
    #     loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")
    #
    #     ctr_accuracy = tf.metrics.accuracy(labels=self.labels['ctr'],
    #                                        predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
    #     cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
    #     ctr_auc = tf.metrics.auc(self.labels['ctr'], ctr_predictions)
    #     cvr_auc = tf.metrics.auc(y, prop)
    #
    #     # ctcvr_auc = tf.metrics.auc(tf.reshape(label_cvr, (-1, 1)), ctcvr_prob)
    #     # ctr_recall = tf.metrics.recall(labels=tf.reshape(label_ctr, (-1, 1)),
    #     #                                predictions=tf.to_float(tf.greater_equal(ctr_prob, 0.5)))
    #     # cvr_recall = tf.metrics.recall(labels=tf.reshape(label_cvr, (-1, 1)),
    #     #                                predictions=tf.to_float(tf.greater_equal(cvr_prob, 0.5)))
    #     # ctcvr_recall = tf.metrics.recall(labels=tf.reshape(label_cvr, (-1, 1)),
    #     #                                  predictions=tf.to_float(tf.greater_equal(ctcvr_prob, 0.5)))
    #     metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    #     tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    #     tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    #     tf.summary.scalar('ctr_auc', ctr_auc[1])
    #     tf.summary.scalar('cvr_auc', cvr_auc[1])
    #     if self.mode == tf.estimator.ModeKeys.EVAL:
    #         return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)
    #
    #     # Create training op.
    #     assert self.mode == tf.estimator.ModeKeys.TRAIN
    #     optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    #     return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)


class export_model(object):
    '''to do'''
    def __init__(self,model=None,input_schema=None,servable_model_dir=None,drop_cols = ['click', 'buy']):
        self.model = model
        self.input_schema = input_schema
        self.servable_model_dir =servable_model_dir
        self.drop_cols = drop_cols

    def export(self):
        feature_spec = get_input_schema_spec(self.input_schema)
        for col in self.drop_cols:
            del feature_spec[col]
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        servable_model_path = self.model.export_savedmodel(self.servable_model_dir, export_input_fn)
        print("*********** Done Exporting at PAth - %s", servable_model_path)




class testw(object):
    '''test case'''
    def __init__(self):
        self.a = 1
    def build_mode(self):
        rand = random.random()
        return rand
        # print(rand)
    def mymodel(self):
        a = self.build_mode()
        b = self.build_mode()
        print(a)
        print(b)
if __name__ == '__main__':
    ss =testw()
    ss.mymodel()