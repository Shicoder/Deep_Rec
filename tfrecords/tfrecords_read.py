# -*- coding: utf-8 -*-
import tensorflow as tf

sess = tf.Session()
dataset = tf.data.TFRecordDataset('./your_path/part-0000.tfrecords')
int_list = ['goods_id','class_id']
float_list = ['pv','uv','age','gender']
string_list = ['search_word']
seq_list = ['search_words']
def parse_tfrecords(x):
    '''并不一定要全部读取，可以随意读取指定index的特征列'''
    feat_s = []
    for filed in float_list:
        a = tf.feature_column.numeric_column(filed)
        feat_s.append(a)
    for filed in float_list:
        a = tf.feature_column.numeric_column(filed)
        feat_s.append(a)
    for filed in string_list:
        a = tf.feature_column.categorical_column_with_hash_bucket(filed,10)
        feat_s.append(a)
    for filed in seq_list:
        a = tf.feature_column.categorical_column_with_hash_bucket(filed, 100)
        feat_s.append(a)

    feature_spec = tf.feature_column.make_parse_example_spec(feat_s)
    feats = tf.parse_single_example(x, features=feature_spec)
    return feats

###############################
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