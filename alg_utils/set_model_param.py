#!/usr/bin/env python
#-*- coding=utf-8 -*-

def set_support(line_num):
    sup = 0.0
    if line_num < 1000:
        sup = 0.05
    if line_num >= 1000 and line_num < 10000:
        sup = 0.003
    if line_num >= 10000 and line_num < 100000:
        sup = 0.0005
    if line_num >= 100000:
        sup = 0.0001
    return sup
