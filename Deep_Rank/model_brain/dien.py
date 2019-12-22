#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:15 PM
'''
import tensorflow as tf
from model_brain import BaseModel
from alg_utils.utils_tf import VecAttGRUCell

from tensorflow.python.ops.rnn_cell import GRUCell
# from tensorflow.python.ops.rnn import dynamic_rnn
from Deep_Rank.model_brain.rnn import dynamic_rnn

class DIEN(BaseModel):
    '''Deep Interest Evolution Network Model'''

    def __init__(self, features, labels, params, mode):
        super(DIEN, self).__init__(features, labels, params, mode)
        # seq feature,
        self.user_aux_loss = True
        self.aux_loss = 0

        self.din_user_goods_seq = features["seq_goods_id_seq"]
        self.din_target_goods_id = features["goods_id"]
        self.goods_embedding_size = 16
        self.goods_bucket_size = 1000
        self.goods_attention_hidden_units = [50, 25]

        self.uid = None

        self.GRU_HIDDEN_SIZE = 8
        self.ATTENTION_SIZE = 8


        # self.din_user_class_seq = features["class_seq"]
        # self.din_target_class_id = features["class_id"]
        _, self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('DIEN_Module'):
            self.logits = self._model_fn

        '''                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue'''

    @property
    def _model_fn(self):
        # history embedding

        din_user_seq = tf.string_to_hash_bucket_fast(self.din_user_goods_seq, self.goods_bucket_size)
        din_target_id = tf.string_to_hash_bucket_fast(self.din_target_goods_id, self.goods_bucket_size)
        name = "goods"
        embedding_size = 8
        bucket_size = 1000

        print(self.din_user_goods_seq.get_shape())
        # batch_size = common_layer.get_shape().as_list()[0]
        # print(batch_size)
        max_his_len = 6
        # 序列长度sequence_length 需要减去值为-1的数，这里先统一成一样的
        # self.sequence_length = [max_his_len]*batch_size
        # self.sequence_length = max_his_len*tf.ones_like(self.labels)
        '''mask为值不为1的长度'''
        # numpy.zeros
        # self.mask = tf.ones_like(self.din_user_goods_seq,dtype=tf.int32)
        self.mask = tf.cast(tf.not_equal(self.din_user_goods_seq, '-1'),dtype=tf.int32)
        self.sequence_length = tf.reduce_sum(self.mask,-1)
        print(self.mask)

        with tf.variable_scope("seq_embedding_table" + name):
            embeddings = self.embedding_table(bucket_size, embedding_size, name)
            seq_emb = tf.nn.embedding_lookup(embeddings,
                                             din_user_seq)  # shape(batch_size, max_seq_len, embedding_size)
            u_his_emb = tf.reshape(seq_emb, shape=[-1, max_his_len,embedding_size])

            tid_emb = tf.nn.embedding_lookup(embeddings, din_target_id)

            if self.user_aux_loss:
                # din_user_seq   [batch_size,max_seq_len]

                tmp_seq = din_user_seq

                tmp_seq = tf.random.shuffle(tmp_seq,seed=1024)
                # tf.batch_gather/tf.gather/tf.gather_nd/
                # 打乱batch索引，用 tf.batch_gather() 取出数据。
                # tf.batch_gather()
                no_seq_emb = tf.nn.embedding_lookup(embeddings,
                                                 tmp_seq)
                # shape(batch_size, max_seq_len, embedding_size)
                self.noclk_u_his_emb = tf.reshape(no_seq_emb, shape=[-1, max_his_len, embedding_size])
                # x = u_his_emb.get_shape().as_list()[1]
                # y = u_his_emb.get_shape().as_list()[2]
                # print("x,y:", x, y)
                # tmp_his_emb = tf.reshape(u_his_emb, [-1, y])
                # tf.random.shuffle(tmp_his_emb)
                # noclk_u_his_emb = tf.reshape(tmp_his_emb, [-1, x, y])

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.GRU_HIDDEN_SIZE), inputs=u_his_emb,
                                         sequence_length=self.sequence_length,
                                         dtype=tf.float32,
                                         scope="gru1")


        if self.user_aux_loss:
            # x = u_his_emb.get_shape().as_list()[1]
            # y = u_his_emb.get_shape().as_list()[2]
            # print("x,y:",x,y)
            # tmp_his_emb = tf.reshape(u_his_emb, [-1, y])
            # tf.random.shuffle(tmp_his_emb)
            # noclk_u_his_emb = tf.reshape(tmp_his_emb, [-1, x, y])
            #
            # print("asas",self.mask.get_shape().as_list())
            # print("dsds", self.din_user_goods_seq.get_shape().as_list())
            #
            # a = self.mask[:, 1:]
            # print("test case:")
            # print(a.get_shape().as_list())
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], u_his_emb[:, 1:, :],
                                             self.noclk_u_his_emb[:, 1:, :],
                                             self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1
            print("auc_loss:",self.aux_loss.get_shape().as_list())

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = self.din_fcn_attention(tid_emb, rnn_outputs, self.ATTENTION_SIZE, self.mask,
                                                         softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(self.GRU_HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)
        # self.uid
        item_his_eb_sum = tf.reduce_sum(u_his_emb, 1)
        inp = tf.concat([self.embedding_layer, tid_emb, item_his_eb_sum, tid_emb * item_his_eb_sum, final_state2], 1)
        print("inp:",inp.get_shape())
        logits = self.fc_net(inp, last_num=1)
        return logits

    def _MY_HEAD(self,
                 mode,
                 label_ctr,
                 ctr_logits):

        ctr_prob = tf.sigmoid(ctr_logits, name="CTR")

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities': ctr_prob
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }

            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)


        ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ctr, logits=ctr_logits),
                                 name='ctr_loss')

        lambda_a = 0.2
        if self.user_aux_loss:
            print("auc_loss:", self.aux_loss.get_shape().as_list())
            print("ctr_loss:", ctr_loss.get_shape().as_list())
            loss = tf.add(ctr_loss, lambda_a * self.aux_loss, name="total_loss")
        else:
            loss = ctr_loss


        ctr_accuracy = tf.metrics.accuracy(labels=label_ctr,
                                           predictions=tf.to_float(tf.greater_equal(ctr_prob, 0.5)))
        ctr_auc = tf.metrics.auc(label_ctr, ctr_prob)
        metrics = {'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc}
        tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
        tf.summary.scalar('ctr_auc', ctr_auc[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
        dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        train_op = dnn_optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @property
    def build_estimator_spec(self):
        '''Build EstimatorSpec'''
        label_ctr = tf.reshape(self.labels, (-1, 1))
        return self._MY_HEAD(
            self.mode,
                 label_ctr,
                 self.logits)
        # if self.user_aux_loss:
        # if self.mode == tf.estimator.ModeKeys.PREDICT:
        #     two_class_logits = tf.concat(
        #         (array_ops.zeros_like(self.logits), self.logits),
        #         axis=-1, name='two_class_logits')
        #     probabilities = tf.nn.softmax(
        #         two_class_logits, name='probabilities')
        #     class_ids = tf.argmax(two_class_logits, axis=-1, name=CLASS_IDS)
        #     class_ids = tf.expand_dims(class_ids, axis=-1)
        #     classes = tf.as_string(class_ids, name='str_classes')
        #     classifier_output = self._classification_output(
        #         scores=probabilities, n_classes=2,
        #         label_vocabulary=None)
        #     predictions = {
        #         'probabilities': probabilities,
        #         CLASS_IDS: class_ids,
        #         CLASSES: classes
        #     }
        #
        #     export_outputs = {
        #         _DEFAULT_SERVING_KEY: classifier_output,
        #         _CLASSIFY_SERVING_KEY: classifier_output,
        #         _PREDICT_SERVING_KEY: tf.estimator.export.PredictOutput(predictions)
        #     }
        #     return tf.estimator.EstimatorSpec(self.mode, predictions=predictions, export_outputs=export_outputs)
        #
        # click_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.reshape(self.labels, (-1, 1)),
        #     logits=self.logits),
        #     name="ctr_loss")
        #
        #
        #
        # accuracy = tf.metrics.accuracy(labels=self.labels['ctr'],
        #                                predictions=tf.to_float(tf.greater_equal(tf.sigmoid(self.logits), 0.5)))
        # auc = tf.metrics.auc(self.labels['ctr'], tf.sigmoid(self.logits))
        #
        # metrics = {'accuracy': accuracy, 'auc': auc}
        # tf.summary.scalar('accuracy', accuracy[1])
        # tf.summary.scalar('auc', auc[1])
        # if self.mode == tf.estimator.ModeKeys.EVAL:
        #     return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)
        #
        # # Create training op.
        # assert self.mode == tf.estimator.ModeKeys.TRAIN
        # optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)
        # else:
        #     my_head = head._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
        #         loss_reduction=losses.Reduction.SUM)
        #     optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        #     return my_head.create_estimator_spec(
        #         features=self.features,
        #         mode=self.mode,
        #         labels=self.labels,
        #         optimizer=optimizer,
        #         logits=self.logits)