#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
@Author: xiangfu shi
@Contact: xfu_shi@163.com
@Time: 2019/12/22 9:18 PM
'''

import tensorflow as tf
from model_brain import BaseModel
class ESMM(BaseModel):
    ''' Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate'''
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
        _, self.Deep_Features = self._get_feature_embedding
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('Din_Module'):
            self.ctr_logits,self.cvr_logits = self._model_fn

    @property
    def _model_fn(self):
        with tf.variable_scope('ctr_model'):
            ctr_logits = self.fc_net(self.embedding_layer)
            # ctr_logits = DIN(self.features, self.labels, self.params, self.mode)
        with tf.variable_scope('cvr_model'):
            cvr_logits = self.fc_net(self.embedding_layer)
            # cvr_logits = DIN(self.features, self.labels, self.params, self.mode)
        return ctr_logits,cvr_logits

    def _MY_HEAD(self,
                 mode,
                 label_ctr,
                 label_cvr,
                 ctr_logits,
                 cvr_logits):

        ctr_prob = tf.sigmoid(self.ctr_logits, name="CTR")
        cvr_prob = tf.sigmoid(self.cvr_logits, name="CVR")
        ctcvr_prob = tf.multiply(ctr_prob, cvr_prob, name="CTCVR")

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities': ctcvr_prob,
                'ctr_probabilities': ctr_prob,
                'cvr_probabilities': cvr_prob
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_ctr, logits=ctr_logits),
                                 name='ctr_loss')
        cvr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cvr, logits=cvr_logits),
                                 name='cvr_loss')
        loss = tf.add(ctr_loss, cvr_loss, name='ctcvr_loss')
        ctr_accuracy = tf.metrics.accuracy(labels=label_ctr,
                                           predictions=tf.to_float(tf.greater_equal(ctr_prob, 0.5)))
        cvr_accuracy = tf.metrics.accuracy(labels=label_cvr, predictions=tf.to_float(tf.greater_equal(cvr_prob, 0.5)))
        ctr_auc = tf.metrics.auc(label_ctr, ctr_prob)
        cvr_auc = tf.metrics.auc(label_cvr, cvr_prob)
        metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
        tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
        tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
        tf.summary.scalar('ctr_auc', ctr_auc[1])
        tf.summary.scalar('cvr_auc', cvr_auc[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
        dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        train_op = dnn_optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @property
    def build_estimator_spec(self):
        label_ctr = tf.reshape(self.labels['ctr'], (-1, 1))
        label_cvr = tf.reshape(self.labels['cvr'], (-1, 1))
        return self._MY_HEAD(
            self.mode,
            label_ctr,
            label_cvr,
            self.ctr_logits,
            self.cvr_logits)
        # ctr_predictions = tf.sigmoid(self.ctr_logits, name="CTR")
        # cvr_predictions = tf.sigmoid(self.cvr_logits, name="CVR")
        # prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
        # price = tf.expand_dims(tf.to_float(self.features["g_goods_price_o"]), axis=-1)
        # ecpm = tf.multiply(prop,tf.log1p(price),name="ECPM")
        # ecpm = self._check_logits_final_dim(ecpm, 1)
        # two_class_ecpm = tf.concat(
        #     (tf.zeros_like(ecpm), ecpm),
        #     axis=-1, name='two_class_logits')
        # class_ids = tf.argmax(two_class_ecpm, axis=-1, name=CLASS_IDS)
        # class_ids = tf.expand_dims(class_ids, axis=-1)
        # print("class_ids shape:", class_ids.shape)
        # classes = tf.as_string(class_ids, name='str_classes')
        # print("classes shape:", classes.shape)
        #
        # if self.mode == tf.estimator.ModeKeys.PREDICT:
        #     predictions = {
        #         'probabilities': ecpm,
        #         # 'ctr_probabilities': ctr_predictions,
        #         # 'cvr_probabilities': cvr_predictions
        #         CLASS_IDS: class_ids,
        #         CLASSES: classes,
        #     }
        #
        #     classifier_output = self._classification_output(
        #         scores=two_class_ecpm, n_classes=2,
        #         label_vocabulary=None)
        #
        #     export_outputs = {
        #     _DEFAULT_SERVING_KEY: classifier_output,
        #     _CLASSIFY_SERVING_KEY: classifier_output,
        #     _PREDICT_SERVING_KEY: tf.estimator.export.PredictOutput(predictions)
        #     }
        #     return tf.estimator.EstimatorSpec(self.mode, predictions=predictions, export_outputs=export_outputs)
        #
        # y = self.labels['cvr']
        # cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(tf.reshape(y,(-1,1)), prop), name="cvr_loss")
        # ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.labels['ctr'],(-1,1)), logits=self.ctr_logits),
        #                          name="ctr_loss")
        # loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")
        #
        # ctr_accuracy = tf.metrics.accuracy(labels=self.labels['ctr'],
        #                                    predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
        # cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
        # ctr_auc = tf.metrics.auc(self.labels['ctr'], ctr_predictions)
        # cvr_auc = tf.metrics.auc(y, prop)
        # metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
        # tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
        # tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
        # tf.summary.scalar('ctr_auc', ctr_auc[1])
        # tf.summary.scalar('cvr_auc', cvr_auc[1])
        # if self.mode == tf.estimator.ModeKeys.EVAL:
        #     return tf.estimator.EstimatorSpec(self.mode, loss=loss, eval_metric_ops=metrics)
        #
        # # Create training op.
        # assert self.mode == tf.estimator.ModeKeys.TRAIN
        # optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # return tf.estimator.EstimatorSpec(self.mode, loss=loss, train_op=train_op)