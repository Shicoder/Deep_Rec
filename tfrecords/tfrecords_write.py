#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import os
import datetime
import tensorflow as tf
from pyspark import SparkContext, HiveContext, SQLContext,SparkConf

def get_n_day_early(the_day, n):
    the_date = datetime.datetime.strptime(str(the_day), u"%Y%m%d")
    the_date -= datetime.timedelta(n)
    early_day = the_date.strftime(u"%Y%m%d")
    return early_day

def create_export_fn(output_path):
    '''定义一个函数，将rdd利用TFRecordWriter写入文件'''
    def export_tfexamples(split_index, iterator):
        path = os.path.join(output_path, 'part-{:04}.tfrecords'.format(split_index))
        print("path_path:",path)
        writer = tf.python_io.TFRecordWriter(path)
        for x in iterator:
            record = x.SerializeToString()
            writer.write(record)
        writer.close()
        yield path
    return export_tfexamples
u'''下面的三个list，里面的值前面不加u有时候会报错'''
int_list = [u'goods_id',u'class_id']
float_list = [u'pv',u'uv',u'age',u'gender']
string_list = [u'search_word']
sequence_list = [u'search_words']

def features_to_tfexample(row):
    '''用于将特征解析成指定的格式，分为整型，浮点型，字符串，序列特征，
       其中如果需要解析序列特征则需要将样本构造成SequenceExample()格式，
       不然的话使用Example()就基本可以满足需求。'''
    feature = {}
    for field in int_list:
        feature[field] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[field])]))
    for field in float_list:
        feature[field] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(row[field])]))
    for field in string_list:
        feature[field] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[row[field].encode('utf-8')]))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    ##################################################
    #     context = tf.train.Features(feature=feature)
    #     feature_list = {}
    #     for field in sequence_list:
    #         feature_list[field] = tf.train.FeatureList(
    #         feature=[tf.train.Feature(float_list=tf.train.FloatList(value=x)) for x in row[field]])
    #
    #     return tf.train.SequenceExample(context=context, feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    return example


if __name__ == '__main__':
    the_day = sys.argv[1]
    range = sys.argv[2]
    mode = sys.argv[3] # train / evaluate /test
    JOB_NAME = u'tfrecords_test'

    ###############################
    download_dir = 'HDFS://your_path/tfrecords/{mode}'.format(mode=mode)
    sc = SparkContext(appName=JOB_NAME)
    ssc = HiveContext(sc)
    # sc.setSystemProperty('spark.driver.maxResultSize', '5g')
    src_sql = '''select * from your_table where par>={start_day} and par<={end_day} and label in (0,1) limit 1000'''.\
        format(start_day=get_n_day_early(the_day,int(range)),end_day=the_day)

    print("step1")
    src_df = ssc.sql(src_sql)
    # src_df = get_dataframe(env,ssc,schema=None,sql_or_path=src_sql).repartition(int(range)+1)
    print("step2")
    example = src_df.rdd.map(lambda row:features_to_tfexample(row))
    print("step3")
    # path = os.path.join(download_dir, 'test2.tf')
    # example.saveAsTextFile(path)
    # path = os.path.join(download_dir, 'test.tf')
    # example.saveAsTextFile(path)
    # example.saveAsNewAPIHadoopFile(download_dir, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
    #                              keyClass="org.apache.hadoop.io.BytesWritable",
    #                              valueClass="org.apache.hadoop.io.NullWritable")
    export_fn = create_export_fn(download_dir)
    files = example.mapPartitionsWithIndex(export_fn).collect()
    print("finished!!")
