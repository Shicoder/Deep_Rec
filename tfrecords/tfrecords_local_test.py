# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import dtypes

######write#########

with tf.python_io.TFRecordWriter('./test.tfrecords') as writer:
    '''定义了三列 g_x g_c pp'''
    for v in [1,10,12]:
        a = ['g_x', 'g_c']
        feature = {}
        for x in a:
            feature[x] = tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

        feature['pp'] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=['test'] ))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    ser_example = example.SerializeToString()
    writer.write(ser_example)

########read#########

sess = tf.Session()
dataset = tf.data.TFRecordDataset('test.tfrecords')
def parse_tfrecords(x):
    '''利用feature_column解析，也可以指定数据类型 FixLengthFeature 来解析，使用feature_column 解析可以很方便的对接到模型'''
    g_x = tf.feature_column.numeric_column('g_x', dtype=dtypes.int64)
    g_c = tf.feature_column.numeric_column('g_c', dtype=dtypes.int64)
    pp = tf.feature_column.categorical_column_with_hash_bucket('pp',1)
    feat_s = [g_x,g_c,pp]
    feature_spec = tf.feature_column.make_parse_example_spec(feat_s)
    feats = tf.parse_single_example(x, features=feature_spec)
    return feats

############run###################
u'''利用dataset api读取数据，高度封装，减少开发难度'''
dataset = dataset.map(parse_tfrecords)
print(dataset)
iterator = dataset.make_one_shot_iterator()
next_data = iterator.get_next()

while True:
    try:
        data = sess.run(next_data)
        print(data)
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break