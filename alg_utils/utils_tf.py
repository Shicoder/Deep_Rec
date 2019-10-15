import json
import tensorflow as tf
def load(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def load_json_from_file(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def get_input_schema_spec(input_schema):
    feature_map = {}
    for input_tensor,param in input_schema.items():
      if param["feature_type"] == "fixed_len":
          type = tf.string if param['value_type'] == 'string' else tf.int64 if  param['value_type'] == 'int' else tf.float32 if param['value_type'] == 'double' else None
          shape = param["value_shape"] if param.has_key("value_shape") else None
          default_value = param["default_value"] if param.has_key("default_value") else None
          if type is None:
              print("no value_type")
          elif shape is not None:
            feature_map[input_tensor] = tf.FixedLenFeature(shape=[int(shape)], dtype=type, default_value=default_value)
          else:
            feature_map[input_tensor] = tf.FixedLenFeature(shape=[], dtype=type, default_value=default_value)
    return feature_map

def PReLU(_x,name=None):
    if name is None:
        name = "alpha"
    _alpha = tf.get_variable(name=name,
                             shape=_x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.25),
                             dtype=_x.dtype)

    return tf.nn.leaky_relu(_x, alpha=_alpha, name=None)