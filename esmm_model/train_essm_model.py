
import sys
import tensorflow as tf
import json
from esmm_model import esmm
from utils.tf_utils import load_json,GET_COLUMNS
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', type=str, default='evaluate',
    help='train/evaluate')
parser.add_argument(
    '--model_dir', type=str, default='hdfs://your_path/model',
    help='Base directory for the model.')
parser.add_argument(
    '--train_epochs', type=int, default=20, help='Number of training epochs.')
parser.add_argument(
    '--batch_size', type=int, default=512, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default="hdfs://your_path/train/part*",
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='hdfs://your_path/test/part*',
    help='Path to the test data.')
parser.add_argument(
    '--servable_model_dir', type=str, default='hdfs://your_path/exported',
    help='Base directory for the eported model.')
parser.add_argument(
    '--profile_dir', type=str, default='hdfs://your_path/profile',
    help='Base directory for the profile model.')



def model_fn(features,
             labels,
             mode,
             params):
  esmm_model = esmm(features,labels,params,mode)
  return esmm_model

def get_feature_spec():
    feature_map = {}
    for fea in input_data:
        type = tf.string if fea['type'] == 'string' or fea['type'] == 'list' else tf.int64 if fea['type'] == 'int' else tf.float32 if \
        fea['type'] == 'double' else None
        if type == None:
            assert "unknown col"
        if fea['name'] == 'goods_id_seq':
            feature_map[fea['name']] = tf.FixedLenFeature(shape=[3],dtype=type, default_value=['-1','-1','-1'])
            continue
        feature_map[fea['name']] = tf.FixedLenFeature(shape=[], dtype=type, default_value=fea['defaultValue'])
    return feature_map
def parse_tfrecords(rows_string_tensor):
  """Takes the string input tensor and returns a dict of rank-2 tensors."""

  feature_spec = get_feature_spec()
  features = tf.parse_single_example(rows_string_tensor, feature_spec)
  label_clcik = features.pop('label_click')
  label_buy = features.pop('label_buy')
  return features,tf.greater_equal(label_clcik,1),tf.greater_equal(label_buy,1)

def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
  files = tf.data.Dataset.list_files(filenames)
  # assert tf.gfile.Exists(filenames), ('%s not found.)
  # dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads,
  #                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # _dataset = _dataset.cache()
  assert files
  dataset = tf.data.TFRecordDataset(files,num_parallel_reads=6)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=40000)

  dataset = dataset.map(parse_tfrecords, num_parallel_calls=6)#num_parallel_calls=tf.data.experimental.AUTOTUNE
  # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  # dataset = dataset.cache()
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  # dataset = dataset.prefetch(40000)
  iterator = dataset.make_one_shot_iterator()
  features, label_click,label_buy = iterator.get_next()
  return features, {"click":tf.to_float(label_click),"buy":tf.to_float(label_buy)}

def main(unused_argv):
    input_index = load_json("inputs_seq.json")
    global input_data
    input_data = input_index['input']['model_name.8.classify']['params']

    WIDE_CATE_COLS, \
    DEEP_EMBEDDING_COLS, \
    CONTINUOUS_COLS, \
    DEEP_SHARED_EMBEDDING_COLS, \
    WIDE_CROSS_COLS \
        = GET_COLUMNS(input_data)

    _HIDDEN_UNITS = [200, 70, 50]
    _DNN_LEARNING_RATE = 0.015

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=600, #save onetime per 300 secs
        keep_checkpoint_max=4 #save lastest 4 checkpoints
    )
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params={
            '_HIDDEN_UNITS': _HIDDEN_UNITS,
            '_DNN_LEARNING_RATE':_DNN_LEARNING_RATE,
            'CONTINUOUS_COLS':CONTINUOUS_COLS,
            'DEEP_EMBEDDING_COLS':DEEP_EMBEDDING_COLS,
            'DEEP_SHARED_EMBEDDING_COLS':DEEP_SHARED_EMBEDDING_COLS
        },
        config= estimator_config)


    '''Generate Timeline'''
    timeline_hook = tf.train.ProfilerHook(save_steps=100000, output_dir=FLAGS.profile_dir, show_dataflow=True,
                                              show_memory=False)

    '''Define train_spec and eval_spec '''
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.train_epochs, True, 0, FLAGS.batch_size), hooks=[timeline_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, 0, FLAGS.batch_size), steps=500,start_delay_secs=300, throttle_secs=300)
    '''Train and evaluate model'''
    results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print(results)

    '''Export Trained Model for Serving'''
    feature_spec = get_feature_spec()
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    servable_model_path = model.export_savedmodel(FLAGS.servable_model_dir, export_input_fn)
    print("*********** Done Exporting at PAth - %s", servable_model_path)
    print("*********** Finshed Total Pipeline ***********")
    return 0

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
