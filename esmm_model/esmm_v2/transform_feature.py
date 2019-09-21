#!/data/venv/hdp-env/bin python
# -*- coding: utf8 -*-
# @Author  : shixiangfu
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import logging
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as fc
from abc import ABCMeta
# from model_fn import ABCMeta

logger = logging.getLogger()


class Error(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class ParametersError(Error):
    def __init__(self, msg):
        super(ParametersError, self).__init__(msg)


class TensorTransform(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    def get_value_tf_type(self, type_name):
        type_value = self.parameters.get(type_name)
        if type_value == "float32":
            return tf.float32
        elif type_value == "float64" or  type_value == "double":
            return tf.float64
        elif type_value == "int8":
            return tf.int8
        elif type_value == "int16":
            return tf.int16
        elif type_value == "int64" or type_value == "int" :
            return tf.int64
        elif type_value == "string":
            return tf.string

        else:
            return None
    def get_default_value(self,type_name):
        type_value = self.parameters.get(type_name)
        if type_value in ("float32","float64","double"):
            return 0.0
        elif type_value in ("int8","int16","int64","int"):
            return 0
        elif type_value == "string":
            return '-1'
        else:
            return None

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        input_tensor = output_tensors.get(input_tensor_name)
        output_tensor_name = self.parameters.get("output_tensor")
        output_tensors[output_tensor_name] = input_tensor


class CateColWithHashBucket(TensorTransform):
    def __init__(self, name, parameters):
        super(CateColWithHashBucket, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        dtype= self.get_value_tf_type("dtype") if self.get_value_tf_type("dtype") != None else tf.string
        if self.parameters.has_key("hash_bucket_size"):
            hash_bucket_size = self.parameters.get("hash_bucket_size")
        else:
            msg = "parameters error, sparse_column_with_hash_bucket must need hash_bucket_size"
            logger.error(msg)
            raise ParametersError(msg)
        print("bucket output_tensor_name:",output_tensor_name)
        output_tensor = fc.categorical_column_with_hash_bucket(
            key=input_tensor_name,
            hash_bucket_size=hash_bucket_size,
            dtype=dtype
        )
        output_tensors[output_tensor_name] = output_tensor


class NumericColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(NumericColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        dtype = self.get_value_tf_type("dtype") if self.get_value_tf_type("dtype") != None else tf.float32
        dv = self.get_default_value("dtype") if self.get_default_value("dtype") != None else 0.0
        if self.parameters.has_key("default_value"):
            default_value = self.parameters.get("default_value")
        else:
            default_value = dv
        output_tensor = fc.numeric_column(
            key = input_tensor_name,
            default_value = default_value,
            dtype = dtype
        )
        print("NumericColumn output_tensor_name:",output_tensor_name)
        output_tensors[output_tensor_name] = output_tensor


class EmbeddingColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(EmbeddingColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("dimension"):
            dimension = self.parameters.get("dimension")
        else:
            msg = "parameters error, embedding_column must need dimension"
            logger.error(msg)
            raise ParametersError(msg)
        input_tensor = output_tensors.get(input_tensor_name)
        ckpt_to_load_from = None
        tensor_name_in_ckpt = None
        if self.parameters.has_key("ckpt_to_load_from") and self.parameters.has_key("tensor_name_in_ckpt"):
            ckpt_to_load_from = self.parameters.get("ckpt_to_load_from")
            tensor_name_in_ckpt = self.parameters.get("tensor_name_in_ckpt")
        combiner = self.parameters.get("combiner") if self.parameters.has_key("combiner") else "mean"
        output_tensor = fc.embedding_column(
            categorical_column = input_tensor,
            dimension = dimension,
            combiner=combiner,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt
        )
        output_tensors[output_tensor_name] = output_tensor


class CateColWithVocabularyList(TensorTransform):
    def __init__(self, name, parameters):
        super(CateColWithVocabularyList, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("vocabulary_list"):
            vocabulary_list = self.parameters.get("vocabulary_list")
            if isinstance(vocabulary_list,list):
                vocabulary_list = tuple(vocabulary_list)
            if isinstance(vocabulary_list,basestring):
                vocabulary_list = tuple(vocabulary_list.split(','))

        else:
            msg = "parameters error, sparse_column_with_keys must need keys"
            logger.error(msg)
            raise ParametersError(msg)


        # combiner = self.parameters.get("combiner", 'sum')
        dtype = self.get_value_tf_type("dtype") if self.get_value_tf_type("dtype") != None else tf.string
        # default_value = self.parameters.get("default_value",None)
        num_oov_buckets = 1

        output_tensors[output_tensor_name] = fc.categorical_column_with_vocabulary_list(
            key = input_tensor_name,
            vocabulary_list = vocabulary_list,
            dtype=dtype,
            # default_value=default_value,
            num_oov_buckets=num_oov_buckets
        )


class CateColWithIdentity(TensorTransform):
    def __init__(self, name, parameters):
        super(CateColWithIdentity, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        default_value = self.parameters.get("default_value", '-1')
        if self.parameters.has_key("bucket_size"):
            bucket_size = self.parameters.get("bucket_size")
        else:
            msg = "parameters error, sparse_column_with_integerized_feature must need bucket_size"
            logger.error(msg)
            raise ParametersError(msg)

        output_tensors[output_tensor_name] = fc.categorical_column_with_identity(
            key=input_tensor_name,
            num_buckets=bucket_size,
            default_value=default_value
        )

class IndicatorColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(IndicatorColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")

        input_tensor = output_tensors.get(input_tensor_name)
        output_tensors[output_tensor_name] = fc.indicator_column(input_tensor)

class BucketizedColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(BucketizedColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("boundaries"):
            boundaries = self.parameters.get("boundaries")
            if not isinstance(boundaries, list):
                boundaries = str(boundaries).replace(' ', '')
                pattern = re.compile('np.linspace\(([0-9]+\.[0-9]+),([0-9]+\.[0-9]+),([0-9]+\.[0-9]+)\)')
                result = pattern.findall(boundaries)
                boundaries = list(np.linspace(float(result[0][0]),
                                              float(result[0][1]),
                                              float(result[0][2])))
        else:
            msg = "parameters error, sparse_column_with_keys must need keys"
            logger.error(msg)
            raise ParametersError(msg)
        print("input_tensor_name:",input_tensor_name)
        input_tensor = output_tensors.get(input_tensor_name)
        output_tensors[output_tensor_name] = fc.bucketized_column(
            source_column=input_tensor,
            boundaries=boundaries)


class SharedEmbeddingColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(SharedEmbeddingColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor = list()
        for input_tensor_name in self.parameters.get("input_tensor"):
            input_tensor.append(output_tensors.get(input_tensor_name))
        if self.parameters.has_key("dimension"):
            dimension = self.parameters.get("dimension")
        else:
            msg = "parameters error, embedding_column must need dimension"
            logger.error(msg)
            raise ParametersError(msg)
        ckpt_to_load_from = None
        tensor_name_in_ckpt = None
        if self.parameters.has_key("ckpt_to_load_from") and self.parameters.has_key("tensor_name_in_ckpt"):
            ckpt_to_load_from = self.parameters.get("ckpt_to_load_from")
            tensor_name_in_ckpt = self.parameters.get("tensor_name_in_ckpt")

        combiner = self.parameters.get("combiner") if self.parameters.has_key("combiner") else "mean"
        shared_embedding_columns = fc.shared_embedding_columns(
            categorical_columns = input_tensor,
            dimension = dimension,
            combiner=combiner,
            ckpt_to_load_from=ckpt_to_load_from,
            tensor_name_in_ckpt=tensor_name_in_ckpt
        )
        for output_tensor_name, output_tensor in zip(self.parameters.get("output_tensor"), shared_embedding_columns):
            output_tensors[output_tensor_name] = output_tensor
            # print("shared_columns:",output_tensor_name,output_tensor)


class CrossedColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(CrossedColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("hash_bucket_size"):
            hash_bucket_size = self.parameters.get("hash_bucket_size")
        else:
            msg = "parameters error, crossed_column must need hash_bucket_size"
            logger.error(msg)
            raise ParametersError(msg)
        column_names = input_tensor_name.split(",")
        columns = []
        for index in range(len(column_names)):
            input_tensor = output_tensors.get(column_names[index])
            columns.append(input_tensor)
        # combiner = self.parameters.get("combiner") if self.parameters.has_key("combiner") else "mean"
        output_tensor = fc.crossed_column(
            keys = columns,
            hash_bucket_size = hash_bucket_size
        )

        output_tensors[output_tensor_name] = output_tensor


class FeatureBuilder(object):
    def __init__(self, model_desc=None):
        self.model_desc = model_desc
        self.output_tensors = dict()

    def process_tensor_transform(self, name, parameters):
        if name == "CateColWithHashBucket":
            tensor_transform = CateColWithHashBucket(name, parameters)
        elif name == "NumericColumn":
            tensor_transform = NumericColumn(name, parameters)
        elif name == "EmbeddingColumn":
            tensor_transform = EmbeddingColumn(name, parameters)
        elif name == "SharedEmbeddingColumn":
            tensor_transform = SharedEmbeddingColumn(name, parameters)
        elif name == "CrossedColumn":
            tensor_transform = CrossedColumn(name, parameters)
        elif name == "CateColWithVocabularyList":
            tensor_transform = CateColWithVocabularyList(name, parameters)
        elif name == "CateColWithIdentity":
            tensor_transform = CateColWithIdentity(name, parameters)
        elif name == "BucketizedColumn":
            tensor_transform = BucketizedColumn(name, parameters)
        elif name == "IndicatorColumn":
            tensor_transform = IndicatorColumn(name, parameters)
        elif name == "":
            tensor_transform = TensorTransform(name, parameters)
        else:
            msg = "transform %s is error or not supported" % name
            logger.error(msg)
            raise ParametersError(msg)
        tensor_transform.transform(self.output_tensors)

    def get_feature_columns(self):
        wide = []
        deep = []
        for tensor_transform in self.model_desc.get("tensorTransform"):
            name = tensor_transform.get("name")
            parameters = tensor_transform.get("parameters")
            print("Process transform %s, input tensor: %s, "
                  "output tensor: %s, wide or deep select: %s,%s"
                  % (name, parameters.get("input_tensor"),
                     parameters.get("output_tensor"),
                     parameters.get("dtype"),
                     parameters.get("wide_or_deep")))
            self.process_tensor_transform(name, parameters)
            wide_or_deep = parameters.get("wide_or_deep")
            if type(parameters.get("output_tensor")) != list:
                parameters["output_tensor"] = [parameters["output_tensor"]]
            for output_tensor_name in parameters.get("output_tensor"):
                out_tensor = self.output_tensors.get(output_tensor_name)
                if not out_tensor:
                    print("Error tensor:",name,parameters.get("output_tensor"))
                    msg = "transform %s process error, output tensor is null" % name
                    logger.error(msg)
                    raise ParametersError(msg)
                if wide_or_deep == "wide":
                    wide.append(out_tensor)
                elif wide_or_deep == "deep":
                    deep.append(out_tensor)
                else:
                    print("column %s not used in wide or deep" % output_tensor_name)


        print("*******************wide columns*******************")
        for i in range(len(wide)):
            print(wide[i])
        print("*******************deep columns*******************")
        for i in range(len(deep)):
            print(deep[i])
        return wide, deep