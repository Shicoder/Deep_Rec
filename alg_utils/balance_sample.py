#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/10 上午10:54
# @Author  : renyanping

#样本均衡函数
def balance(positive_data,negative_data):

    positive_count= positive_data.count()
    negative_count = negative_data.count()

    weights = []

    if positive_count > negative_count:
        times = positive_count/negative_count
        for i in range(0, times):
            weights.append(1.0)
        sample_positive_data = positive_data.randomSplit(weights)[0]
        sample_data = sample_positive_data.unionAll(negative_data)

    else:
        times = negative_count/positive_count
        for i in range(0, times):
            weights.append(1.0)
        sample_negative_data = negative_data.randomSplit(weights)[0]
        sample_data = positive_data.union(sample_negative_data)

    print "样本总数：",sample_data.count()
    return sample_data

#训练集和测试集划分函数
def split_train_test(raw_rdd, split_ratio):
    return raw_rdd.randomSplit([split_ratio, 1 - split_ratio])