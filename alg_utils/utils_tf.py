import json
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops.rnn_cell_impl import RNNCell
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

def PReLU2( _x, name=None):
    if name is None:
        name = "alpha"
    _alpha = tf.get_variable(name,
                                shape=_x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.25),
                                dtype=_x.dtype)

    return tf.maximum(_alpha * _x, _x)

def dice(_x, axis=-1, epsilon=0.000000001, name='dice'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    broadcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - broadcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    broadcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - broadcast_mean) / (broadcast_std + epsilon)
    x_p = tf.sigmoid(x_normed)
    return alphas * (1.0 - x_p) * _x + x_p * _x



class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h












