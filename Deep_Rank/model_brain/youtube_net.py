#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
Author  : xiangfu shi
Email   : xfu_shi@163.com
'''
import tensorflow as tf
import sys
from operator import mul
sys.path.append("..")
from tensorflow.python.saved_model import signature_constants
import random
from model_brain import  BaseModel

CLASSES = 'classes'
CLASS_IDS = 'class_ids'

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
_CLASSIFY_SERVING_KEY = 'classification'
_PREDICT_SERVING_KEY = 'predict'
_REGRESS_SERVING_KEY = 'regression'

class youtube_net(BaseModel):
    '''
    youtube 出的排序模型
    "Recommending What Video to Watch Next: A Multitask Ranking System"
    '''
    def __init__(self, features, labels, params, mode):
        super(youtube_net,self).__init__(features, labels, params, mode)
        self.Shallow_Features, \
        self.Deep_Features = self._get_feature_embedding
        self.num_tasks = 2
        self.tasks = ['ctr', 'cvr']
        with tf.variable_scope('Embedding_Module'):
            self.embedding_layer = self.get_input_layer(self.Deep_Features)
        with tf.variable_scope('MMoE_Module'):
            self.logits = self._model_fn


    def mmoe_net(self,net,units = 8,experts_num = 2):
        gate_outputs = []
        final_outputs = []

        input_dim = net.get_shape().as_list()[1]
        expert_kernels = tf.get_variable(
            name="expert_kernels",
            shape=(input_dim,units,experts_num),
            initializer=tf.initializers.variance_scaling(),
            )
        # use_expert_bias
        gate_kernels = [tf.get_variable(
            name="gate_kernels_{}".format(str(i)),
            shape=(input_dim,experts_num),
            initializer=tf.initializers.variance_scaling()) for i in range(self.num_tasks)]

        # use_gate_bias

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.tensordot(a=net,b=expert_kernels,axes=1)
        expert_outputs = tf.nn.relu(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index,gate_kernels in enumerate(gate_kernels):
            print("net:",net.get_shape().as_list())
            print("gate_kernels:", gate_kernels.get_shape().as_list())
            gate_output = tf.tensordot(a = net ,b = gate_kernels,axes=1)
            print("gate_output:",gate_output.get_shape().as_list())
            # user_fate_bias
            gate_output = tf.nn.softmax(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output,axis=1)
            weighted_expert_output = expert_outputs*tf.keras.backend.repeat_elements(expanded_gate_output,units,axis=1)
            final_outputs.append(tf.reduce_sum(weighted_expert_output,axis=2))

        return final_outputs


    @property
    def _model_fn(self):
        # context_scene
        output_layers = []
        self.shallow_tower_layer = self.get_input_layer(self.Shallow_Features)
        self.shallow_tower_logit = tf.layers.dense(self.shallow_tower_layer, units=1)

        mmoe_layers = self.mmoe_net(self.embedding_layer,units=8,experts_num=2)
        for index ,task_layer in enumerate(mmoe_layers):
            tower_layer = tf.layers.dense(task_layer,units=5,activation='relu',kernel_initializer=tf.initializers.variance_scaling())
            output_layer = tf.layers.dense(tower_layer,units=1,name='logits_{}'.format(str(index)),kernel_initializer=tf.initializers.variance_scaling())
            # logit = tf.sigmoid(tf.add(output_layer,self.shallow_tower_logit))
            logit = tf.add(output_layer, self.shallow_tower_logit)
            output_layers.append(logit)

        return output_layers


    def _MY_HEAD(self,
                    mode,
                    labels,
                    logits):
        probs = [tf.sigmoid(logit) for logit in logits]
        losses = []
        tmp = {}
        for name,i in zip(self.tasks,range(len(logits))):
            losses.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(labels[name],shape=(-1,1)), logits=logits[i])))
            key = "{}_probabilities".format(name)
            value = probs[i]
            tmp[key] = value



        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities': reduce(mul,probs)
            }
            predictions.update(tmp)

            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        loss = tf.reduce_sum(losses)

        metrics = {}
        for name, i in zip(self.tasks, range(len(logits))):
            accuracy = tf.metrics.accuracy(labels=labels[name],
                                           predictions=tf.to_float(tf.greater_equal(probs[i], 0.5)))
            auc = tf.metrics.auc(labels[name], probs[i])
            key_accuracy = '{}_accuracy'.format(name)
            key_auc = '{}_auc'.format(name)
            metrics[key_accuracy] = accuracy
            metrics[key_auc] = auc
            tf.summary.scalar('{}_accuracy'.format(name), accuracy[1])
            tf.summary.scalar('{}_auc'.format(name), auc[1])


        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
        dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=self.params['LEARNING_RATE'])
        train_op = dnn_optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @property
    def build_estimator_spec(self):
        return self._MY_HEAD(
            self.mode,
            self.labels,
            self.logits)

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