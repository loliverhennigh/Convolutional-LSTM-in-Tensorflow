
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

def _variable(name, shape, initializer):
  """Helper to create a Variable.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # getting rid of stddev for xavier ## testing this for faster convergence
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable('weights', shape=[kernel_size,kernel_size,input_channels,num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv = nonlinearity(conv)
    return conv

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable('weights', shape=[kernel_size,kernel_size,num_features,input_channels],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = _variable('biases',[num_features],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    if nonlinearity is not None:
      conv = nonlinearity(conv)
    shape = int_shape(inputs)
    conv = tf.reshape(conv, [shape[0], shape[1]*stride, shape[2]*stride, num_features])
    return conv

def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable('weights', shape=[dim,hiddens],initializer=tf.contrib.layers.xavier_initializer())
    biases = _variable('biases', [hiddens], initializer=tf.contrib.layers.xavier_initializer())
    output_biased = tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    if nonlinearity is not None:
      output_biased = nonlinearity(ouput_biased)
    return output_biased

def nin(x, num_units, idx):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = fc_layer(x, num_units, idx)
    return tf.reshape(x, s[:-1]+[num_units])

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=True, name="resnet", begin_nonlinearity=True):
      
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

def res_block_lstm(x, hidden_state=None, keep_p=1.0, name="resnet_lstm"):

  orig_x = x
  if hidden_state is not None:
    hidden_state_1 = hidden_state[0]
    hidden_state_2 = hidden_state[1]
  else:
    hidden_state_1 = None
    hidden_state_2 = None
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

  return orig_x + x_2, [hidden_state_1, hidden_state_2]
