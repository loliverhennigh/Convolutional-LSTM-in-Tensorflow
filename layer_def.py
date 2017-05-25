
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np

import BasicConvLSTMCell 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)

def int_shape(x):
  return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def set_nonlinearity(name):
  if name == 'concat_elu':
    return concat_elu
  elif name == 'elu':
    return tf.nn.elu
  elif name == 'concat_relu':
    return tf.nn.crelu
  elif name == 'relu':
    return tf.nn.relu
  else:
    raise('nonlinearity ' + name + ' is not supported')

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=False):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv = nonlinearity(conv)
    return conv_rect

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, linear = False):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.01))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect
     

def fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=FLAGS.weight_init, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(FLAGS.weight_init))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.elu(ip,name=str(idx)+'_fc')

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet", begin_nonlinearity=True):
      
  orig_x = x
  if begin_nonlinearity: 
    x = nonlinearity(x) 
  if stride == 1:
    x = conv_layer(x, 3, stride, filter_size, name + '_conv_1')
  elif stride == 2:
    x = conv_layer(x, 4, stride, filter_size, name + '_conv_1')
  else:
    print("stride > 2 is not supported")
    exit()
  if a is not None:
    shape_a = int_shape(a) 
    shape_x_1 = int_shape(x)
    a = tf.pad(
      a, [[0, 0], [0, shape_x_1[1]-shape_a[1]], [0, shape_x_1[2]-shape_a[2]],
      [0, 0]])
    x += nin(nonlinearity(a), filter_size, name + '_nin')
  x = nonlinearity(x)
  if keep_p < 1.0:
    x = tf.nn.dropout(x, keep_prob=keep_p)
  if not gated:
    x = conv_layer(x, 3, 1, filter_size, name + '_conv_2')
  else:
    x = conv_layer(x, 3, 1, filter_size*2, name + '_conv_2')
    x_1, x_2 = tf.split(x,2,3)
    x = x_1 * tf.nn.sigmoid(x_2)

  if int(orig_x.get_shape()[2]) > int(x.get_shape()[2]):
    assert(int(orig_x.get_shape()[2]) == 2*int(x.get_shape()[2]), "res net block only supports stirde 2")
    orig_x = tf.nn.avg_pool(orig_x, [1,2,2,1], [1,2,2,1], padding='SAME')

  # pad it
  out_filter = filter_size
  in_filter = int(orig_x.get_shape()[-1])
  if out_filter > in_filter:
    orig_x = tf.pad(
        orig_x, [[0, 0], [0, 0], [0, 0],
        [(out_filter-in_filter), 0]])
  elif out_filter < in_filter:
    orig_x = nin(orig_x, out_filter, name + '_nin_pad')
  return orig_x + x

def res_block_lstm(x, hidden_state_1=None, hidden_state_2=None, keep_p=1.0, name="resnet_lstm"):

  orig_x = x
  filter_size = orig_x.get_shape().as_list()[-1]

  with tf.variable_scope(name + "_conv_LSTM_1", initializer = tf.random_uniform_initializer(-0.01, 0.01)) as scope:
    lstm_cell_1 = BasicConvLSTMCell.BasicConvLSTMCell([int(x.get_shape()[1]),int(x.get_shape()[2])], [3,3], filter_size)
    if hidden_state_1 == None:
      batch_size = x.get_shape()[0]
      hidden_state_1 = lstm_cell_1.zero_state(batch_size, tf.float32) 
    x_1, hidden_state_1 = lstm_cell_1(x, hidden_state_1, scope=scope)
    
  if keep_p < 1.0:
    x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)

  with tf.variable_scope(name + "_conv_LSTM_2", initializer = tf.random_uniform_initializer(-0.01, 0.01)) as scope:
    lstm_cell_2 = BasicConvLSTMCell.BasicConvLSTMCell([int(x_1.get_shape()[1]),int(x_1.get_shape()[2])], [3,3], filter_size)
    if hidden_state_2 == None:
      batch_size = x_1.get_shape()[0]
      hidden_state_2 = lstm_cell_2.zero_state(batch_size, tf.float32) 
    x_2, hidden_state_2 = lstm_cell_2(x_1, hidden_state_2, scope=scope)

  return orig_x + x_2, hidden_state_1, hidden_state_2
