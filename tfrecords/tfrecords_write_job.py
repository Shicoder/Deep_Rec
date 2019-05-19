#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-

import os
from os.path import abspath, dirname
now_real_path=dirname(abspath(__file__)) + "/"
import sys
import datetime

SPARK_HOME="your spark home"
JOB_HOME = now_real_path
REAL_JOB_PATH = JOB_HOME + "tfrecords_write.py"

LOG_PATH = now_real_path + "log_tf.txt"
EGG_PATH = now_real_path + "dist/tfrecords.egg"
the_day = sys.argv[1]
range = sys.argv[2]
mode = sys.argv[3]
def run_job():
    cmd = '''%s/bin/spark-submit --master yarn-client --queue default --executor-memory 4g --num-executors 32 --conf spark.yarn.executor.memoryOverhead=1024 --conf spark.executorEnv.LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server:$LIB_CUDA:/usr/local/hadoop/lib0 --conf spark.executorEnv.CLASSPATH=$(hadoop classpath --glob) --py-files %s %s %s %s %s''' % \
          (SPARK_HOME,EGG_PATH, REAL_JOB_PATH,the_day,range,mode)
    print(cmd)
    res = os.system(cmd)
    if res != 0:
        print("run spark job failed")
        sys.exit(-1)
def run():
    run_job()

if __name__=="__main__":
    run()
# from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
# build_raw_serving_input_receiver_fn
'''运行脚本参考：
python tfrecords_write_job.py 20190418 5 train
将数据保存到Hadoop必须加入两个配置参数：
spark.executorEnv.LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server:$LIB_CUDA:/usr/local/hadoop/lib0
spark.executorEnv.CLASSPATH=$(hadoop classpath --glob)
'''