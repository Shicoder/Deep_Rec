#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
'''
Author  : xiangfu shi
Email   : xfu_shi@163.com
'''
# 啊
import sys
sys.path.append("..")
import tensorflow as tf
from alg_utils.utils_tf import load_json_from_file,get_input_schema_spec
from model_brain.model_brain import export_model
from Deep_Rank.model_brain.dcn import DCN
from Deep_Rank.model_brain.dnn_demo import DNN
from Deep_Rank.model_brain.deepFM import DeepFM
from Deep_Rank.model_brain.dien import DIEN
from Deep_Rank.model_brain.din import DIN
from Deep_Rank.model_brain.dssm import DSSM
from Deep_Rank.model_brain.esmm import ESMM
from Deep_Rank.model_brain.wide_deep import WD_Model
from Deep_Rank.model_brain.xDeepFM import xDeepFM
'''nohup python model.py > log 2>&1 &'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', type=str, default='evaluate',
    help='train/evaluate')
parser.add_argument(  #hdfs://your_path/model
    '--model_dir', type=str, default='../model',
    help='Base directory for the model.')
parser.add_argument(
    '--train_epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument(
    '--batch_size', type=int, default=1024, help='Number of examples per batch.')
parser.add_argument(#hdfs://your_path//train/part*
    '--train_data', type=str, default="../data/train/part*",
    help='Path to the training data.')
parser.add_argument(#hdfs://your_path/test/part*
    '--test_data', type=str, default='../data/test/part*',
    help='Path to the test data.')
parser.add_argument(
    '--servable_model_dir', type=str, default='../exported',
    help='Base directory for the eported model.')
parser.add_argument(
    '--profile_dir', type=str, default='../profile',
    help='Base directory for the eported model.')
parser.add_argument(
    '--is_profile', type=bool, default=False, help='if true ,open profile')
parser.add_argument(
    '--model_name', type=str, default='xdeepfm',
    help='model')

def model_fn(features,
             labels,
             mode,
             params):
  '''model_fn'''
  model = None
  model_name = FLAGS.model_name
  if model_name == 'dnn':
      model = DNN(features, labels, params, mode)
  elif model_name == 'dcn':
      model = DCN(features, labels, params, mode)
  elif model_name == 'wd':
      model = WD_Model(features, labels, params, mode)
  elif model_name == 'din':
      model = DIN(features, labels, params, mode)
  elif model_name == 'esmm':
      model = ESMM(features, labels, params, mode)
  elif model_name == 'deepfm':
      model = DeepFM(features, labels, params, mode)
  elif model_name == 'dien':
      model = DIEN(features, labels, params, mode)
  elif model_name == 'dssm':
      model = DSSM(features, labels, params, mode)
  elif model_name == 'xdeepfm':
      model = xDeepFM(features, labels, params, mode)
  # 2
  elif model_name == 'pnn':
      # model = PNN(features, labels, params, mode)
      pass
  # elif model_name == 'dssm':
  #     model = DSSM(features, labels, params, mode)
  # elif model_name == 'bilinear':
  #     model = BiLinear(features, labels, params, mode)
  # elif model_name == 'DSIN':
  #     model = DSIN(features, labels, params, mode)
  # 3
  elif model_name == 'bst':
      # model = BST(features, labels, params, mode)
      pass
  # 4
  elif model_name == 'irgan':
      # model = IRGAN(features, labels, params, mode)
      pass
  estimator_spec = model.build_estimator_spec

  return estimator_spec


def parse_tfrecords(rows_string_tensor):
  '''parse_tfrecords'''
  input_cols = get_input_schema_spec(input_schema)
  features = tf.parse_single_example(rows_string_tensor, input_cols)
  if use_esmm:
      label_clcik = features.pop('label_click')
      label_buy = features.pop('label_buy')
      return features, tf.greater_equal(label_clcik, 1), tf.greater_equal(label_buy, 1)
  else:
      label_clcik = features.pop('label_click')
      label_buy = features.pop('label_buy')
      return features, tf.greater_equal(label_clcik, 1),0

  # label_clcik = features.pop('click')
  # label_buy = features.pop('buy')


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             batch_size=200):
  '''input_fn'''
  files = tf.data.Dataset.list_files(filenames)
  assert files
  dataset = tf.data.TFRecordDataset(files,num_parallel_reads=6)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=40000)

  dataset = dataset.map(parse_tfrecords, num_parallel_calls=6)#num_parallel_calls=tf.data.experimental.AUTOTUNE
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features, label_click,label_buy = iterator.get_next()

  if use_esmm:
      return features, {"ctr":tf.to_float(label_click),"cvr":tf.to_float(label_buy)}
  else:
      return features, tf.to_float(label_click)


input_schema = load_json_from_file("./model_schema.json")["schema"]
model_feature = load_json_from_file("./model_feature.json.deepfm")
def main(unused_argv):

    global use_esmm
    if FLAGS.model_name == 'esmm':
        use_esmm = True
    else:
        use_esmm = False

    _HIDDEN_UNITS = [200, 70, 50]
    _DNN_LEARNING_RATE = 0.002
    _LINEAR_LEARNING_RATE = 0.0001
    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=600, #save onetime per 300 secs
        keep_checkpoint_max=4 #save lastest 4 checkpoints
    )
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'HIDDEN_UNITS': _HIDDEN_UNITS,
            'LEARNING_RATE':_DNN_LEARNING_RATE,
            'LINEAR_LEARNING_RATE':_LINEAR_LEARNING_RATE,
            'FEATURES_DICT':model_feature,
            'CROSS_LAYER_NUM':2 #dcn
        },
        config= estimator_config)
    '''Generate Timeline'''
    timeline_hook =None
    if FLAGS.is_profile:
        timeline_hook = tf.train.ProfilerHook(save_steps=100000, output_dir=FLAGS.profile_dir, show_dataflow=True,
                                              show_memory=False)
    '''Train and Evaluate,Define train_spec and eval_spec '''
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.train_epochs, True, FLAGS.batch_size), hooks=[timeline_hook] if FLAGS.is_profile else None)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size), steps=800,start_delay_secs=300, throttle_secs=300)
    '''Train and evaluate model'''
    results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print(results)
    '''Export Trained Model for Serving'''
    export_path = export_model(model,input_schema,FLAGS.servable_model_dir,drop_cols=['label_click', 'label_buy'])
    print(export_path)
    print("*********** Finshed Total Pipeline ***********")

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


