#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
# @Author  : shixiangfu
import sys
import os
from os.path import abspath, dirname
now_real_path=dirname(abspath(__file__)) + "/"
JOB_HOME = now_real_path
print("now_real_path:",now_real_path)

REAL_JOB_PATH = JOB_HOME + "train_esmm_model.py"
LOG_PATH = now_real_path + "log_tf.txt"

def run_tf_hdfs_job():
    cmd_1 = '''hadoop fs -rm -r -f hdfs://my_path/model/'''
    print(cmd_1)
    res = os.system(cmd_1)
    if res != 0:
        print("run spark job failed,code is :",cmd_1)
        sys.exit(-1)

    cmd = '''LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hadoop/lib/native CLASSPATH=$(hadoop classpath --glob) /data/venv/hdp-env/bin/python %s '''% \
          (REAL_JOB_PATH)
    print(cmd)
    res = os.system(cmd)
    if res != 0:
        print("run spark job failed,code is :",cmd)
        sys.exit(-1)

def run():
    run_tf_hdfs_job()

if __name__=="__main__":
    run()