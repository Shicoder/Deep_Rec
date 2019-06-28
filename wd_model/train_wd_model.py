# export HADOOP_HOME=/tmp/user/hadoop-2.7
# export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)

import multiprocessing
import sys
import tensorflow as tf
import json
from wide_and_deep_model import wide_and_deep

'''nohup python model.py > wd_log 2>&1 &'''
import argparse
# from alg_utils.utils_tf import load
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', type=str, default='evaluate',
    help='train/evaluate')
parser.add_argument(
    '--model_dir', type=str, default='hdfs://your_path/model',
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default='wide',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=1, help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--step_for_eval', type=int, default=500,
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=400, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default="hdfs://your_path/train/part*",
    help='Path to the training data.')
parser.add_argument(
    '--test_data', type=str, default='hdfs://your_path/test/part*',
    help='Path to the test data.')
parser.add_argument(
    '--servable_model_dir', type=str, default='hdfs://your_path/youzan_exported',
    help='Base directory for the eported model.')
parser.add_argument(
    '--profile_dir', type=str, default='hdfs://your_path/youzan_profile',
    help='Base directory for the eported model.')
def load(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data
input_index = load("../config/inputs_v3_seq.json")
input_data = input_index['input']['model_name.classify']['params']

WIDE_CATE_COLS = []
DEEP_EMBEDDING_COLS = []
CONTINUOUS_COLS = []
DEEP_SHARED_EMBEDDING_COLS = []

ORIGIN_DEEP_SHARED_EMBEDDING_COLS = []
for fea in input_data:
    if 'col_type' in fea:
        if type(fea['col_type']).__name__ == 'list':
            for col in fea['col_type']:
                if col == 'WIDE_CATE_COLS':
                    WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
                if col == 'DEEP_EMBEDDING_COLS':
                    DEEP_EMBEDDING_COLS.append((fea['name'], fea['bucket_size'], fea['embedding_size'],fea['type']))
                if col == 'CONTINUOUS_COLS':
                    CONTINUOUS_COLS.append(fea['name'])
                if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                    ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append(
                        (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type'], fea['shared_flag']))
        else:
            if fea['col_type'] == 'WIDE_CATE_COLS':
                WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
            if fea['col_type'] == 'DEEP_EMBEDDING_COLS':
                DEEP_EMBEDDING_COLS.append((fea['name'], fea['bucket_size'], fea['embedding_size'],fea['type']))
            if fea['col_type'] == 'CONTINUOUS_COLS':
                CONTINUOUS_COLS.append(fea['name'])
            if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append((fea['name'], fea['bucket_size'], fea['embedding_size'],fea['type'],fea['shared_flag']))
print("ORIGIN_DEEP_SHARED_EMBEDDING_COLS:",ORIGIN_DEEP_SHARED_EMBEDDING_COLS)
shared_flags = set()
for _,_,_,_,flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
    shared_flags.add(flag)

for c_flag in shared_flags:
    names = []
    bucket_sizes = []
    embedding_sizes = []
    types = []
    for name, bucket_size, embedding_size, type, flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
        if c_flag==flag:
            names.append(name)
            bucket_sizes.append(bucket_size)
            embedding_sizes.append(embedding_size)
            types.append(type)
    DEEP_SHARED_EMBEDDING_COLS.append((names,bucket_sizes[0],embedding_sizes[0],types[0],c_flag))




print("DEEP_SHARED_EMBEDDING_COLS:",DEEP_SHARED_EMBEDDING_COLS)
print("WIDE_CATE_COLS:",WIDE_CATE_COLS)
print('CONTINUOUS_COLS:',CONTINUOUS_COLS)
print('DEEP_EMBEDDING_COLS:',DEEP_EMBEDDING_COLS)


WIDE_CROSS_COLS = (('age','class1',140),('age','class2',140))



def model_fn(mode,
             features,
             labels,params):

  my_head,logits,_train_op_fn = wide_and_deep(features,params)
  return my_head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      logits=logits,
      # train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
      train_op_fn=_train_op_fn
  )

def get_feature_spec():
    feature_map = {}
    for fea in input_data:
        type = tf.string if fea['type'] == 'string' or fea['type'] == 'seq' else tf.int64 if fea['type'] == 'int' else tf.float32 if \
        fea['type'] == 'double' else None
        if type == None:
            assert "unknown col"
        if fea['name'] == 'goods_id_seq':
            feature_map[fea['name']] = tf.FixedLenFeature(shape=[3],dtype=type, default_value=['-1','-1','-1'])
            continue
        feature_map[fea['name']] = tf.FixedLenFeature(shape=[], dtype=type, default_value=fea['defaultValue'])
    return feature_map
def parse_tfrecords(rows_string_tensor):

  feature_spec = get_feature_spec()
  features = tf.parse_single_example(rows_string_tensor, feature_spec)
  label = features.pop('label')
  return features,tf.greater_equal(label,1)

def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             batch_size=200):
  files = tf.data.Dataset.list_files(filenames)
  assert files
  dataset = tf.data.TFRecordDataset(files,num_parallel_reads=2)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=40000)

  dataset = dataset.map(parse_tfrecords, num_parallel_calls=2)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels

def main(unused_argv):

    _HIDDEN_UNITS = [150, 70, 50]
    _DNN_LEARNING_RATE = 0.001
    _LINEAR_LEARNING_RATE = 0.004

    estimator_config = tf.estimator.RunConfig(
        save_checkpoints_secs=600, #save onetime per 300 secs
        keep_checkpoint_max=4 #save lastest 4 checkpoints
    )
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params={
            '_HIDDEN_UNITS': _HIDDEN_UNITS,
            '_LINEAR_LEARNING_RATE': _LINEAR_LEARNING_RATE,
            '_DNN_LEARNING_RATE':_DNN_LEARNING_RATE,
            'WIDE':True,
            'DEEP':True,
            'WIDE_CATE_COLS':WIDE_CATE_COLS,
            'CONTINUOUS_COLS':CONTINUOUS_COLS,
            'DEEP_EMBEDDING_COLS':DEEP_EMBEDDING_COLS,
            'WIDE_CROSS_COLS':WIDE_CROSS_COLS,
            'DEEP_SHARED_EMBEDDING_COLS':DEEP_SHARED_EMBEDDING_COLS
        },
        config= estimator_config)


    '''Generate Timeline'''
    timeline_hook = tf.train.ProfilerHook(save_steps=100000, output_dir=FLAGS.profile_dir, show_dataflow=True,
                                              show_memory=False)

    '''Define train_spec and eval_spec '''
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.train_epochs, True, FLAGS.batch_size), hooks=[timeline_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size), steps=800,start_delay_secs=600, throttle_secs=600)
    '''Train and evaluate model'''
    results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print(results)

    '''Export Trained Model for Serving'''
    # wideColumns, DeepColumns = build_model_columns()
    # feature_columns = wideColumns + DeepColumns
    # varlenfeature
    # "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
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
