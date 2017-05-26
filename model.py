
import BasicConvLSTMCell
from layer_def import *



def select_network(name="residual_u_network"):
  network = None
  if name == "basic_network":
    network = basic_network_template 
  elif name == "residual_u_network":
    network = residual_u_network_template
  return network

def basic_network(inputs, hidden, lstm=True):
  conv1 = conv_layer(inputs, 3, 2, 8, "encode_1", nonlinearity=tf.nn.elu)
  # conv2
  conv2 = conv_layer(conv1, 3, 1, 8, "encode_2", nonlinearity=tf.nn.elu)
  # conv3
  conv3 = conv_layer(conv2, 3, 2, 8, "encode_3", nonlinearity=tf.nn.elu)
  # conv4
  conv4 = conv_layer(conv3, 1, 1, 4, "encode_4", nonlinearity=tf.nn.elu)
  y_0 = conv4
  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([8,8], [3,3], 4)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_1, hidden = cell(y_0, hidden)
  else:
    y_1 = conv_layer(y_0, 3, 1, 8, "encode_3", nonlinearity=tf.nn.elu)
 
  # conv5
  conv5 = transpose_conv_layer(y_1, 1, 1, 8, "decode_5", nonlinearity=tf.nn.elu)
  # conv6
  conv6 = transpose_conv_layer(conv5, 3, 2, 8, "decode_6", nonlinearity=tf.nn.elu)
  # conv7
  conv7 = transpose_conv_layer(conv6, 3, 1, 8, "decode_7", nonlinearity=tf.nn.elu)
  # x_1 
  x_1 = transpose_conv_layer(conv7, 3, 2, 3, "decode_8") # set activation to linear

  return x_1, hidden

# make a template for reuse
basic_network_template = tf.make_template('basic_network', basic_network)

def residual_u_network(inputs, hiddens=None, start_filter_size=16, nr_downsamples=3, nr_residual_per_downsample=1, nonlinearity="concat_elu"):

  # set filter size (after each down sample the filter size is doubled)
  filter_size = start_filter_size

  # set nonlinearity
  nonlinearity = set_nonlinearity(nonlinearity)

  # make list of hiddens if None
  if hiddens is None:
    hiddens = (2*nr_downsamples -1)*[None]

  # store for u network connections and new hiddens
  a = []
  hidden_out = []

  # encoding piece
  x_i = inputs
  for i in xrange(nr_downsamples):
    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
    for j in xrange(nr_residual_per_downsample - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    x_i, hidden_new = res_block_lstm(x_i, hiddens[i], name="res_encode_lstm_" + str(i))
    a.append(x_i)
    hidden_out.append(hidden_new)
    filter_size = filter_size * 2

  # pop off last element to a.
  a.pop()
  filter_size = filter_size / 2

  # decoding piece
  for i in xrange(nr_downsamples - 1):
    filter_size = filter_size / 2
    x_i = transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_per_downsample):
      x_i = res_block(x_i, a=a.pop(), filter_size=filter_size, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    x_i, hidden_new = res_block_lstm(x_i, hiddens[i + nr_downsamples], name="res_decode_lstm_" + str(i))
    hidden_out.append(hidden_new)

  x_i = transpose_conv_layer(x_i, 4, 2, int(inputs.get_shape()[-1]), "up_conv_" + str(nr_downsamples-1))

  return x_i, hidden_out 

# make template for reuse
residual_u_network_template = tf.make_template('residual_u_network', residual_u_network)


    

