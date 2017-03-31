
import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 

def network(inputs, hidden, lstm=True):
  conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
  # conv3
  conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
  y_0 = conv4
  if lstm:
    # conv lstm cell 
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell.BasicConvLSTMCell([8,8], [3,3], 4)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_1, hidden = cell(y_0, hidden)
  else:
    y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")
 
  # conv5
  conv5 = ld.transpose_conv_layer(y_1, 1, 1, 8, "decode_5")
  # conv6
  conv6 = ld.transpose_conv_layer(conv5, 3, 2, 8, "decode_6")
  # conv7
  conv7 = ld.transpose_conv_layer(conv6, 3, 1, 8, "decode_7")
  # x_1 
  x_1 = ld.transpose_conv_layer(conv7, 3, 2, 3, "decode_8", True) # set activation to linear

  return x_1, hidden

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 32, 32, 3])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    # conv network
    hidden = None
    for i in xrange(FLAGS.seq_length-1):
      if i < FLAGS.seq_start:
        x_1, hidden = network_template(x_dropout[:,i,:,:,:], hidden)
      else:
        x_1, hidden = network_template(x_1, hidden)
      x_unwrap.append(x_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])

    # this part will be used for generating video
    x_unwrap_g = []
    hidden_g = None
    for i in xrange(50):
      if i < FLAGS.seq_start:
        x_1_g, hidden_g = network_template(x_dropout[:,i,:,:,:], hidden_g)
      else:
        x_1_g, hidden_g = network_template(x_1_g, hidden_g)
      x_unwrap_g.append(x_1_g)

    # pack them generated ones
    x_unwrap_g = tf.stack(x_unwrap_g)
    x_unwrap_g = tf.transpose(x_unwrap_g, [1,0,2,3,4])

    # calc total loss (compare x_t to x_t+1)
    loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - x_unwrap[:,FLAGS.seq_start:,:,:,:])
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, 32, FLAGS.num_balls)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        # make video
        print("now generating video!")
        video = cv2.VideoWriter()
        success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
        dat_gif = dat
        ims = sess.run([x_unwrap_g],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
        ims = ims[0][0]
        print(ims.shape)
        for i in xrange(50 - FLAGS.seq_start):
          x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
          new_im = cv2.resize(x_1_r, (180,180))
          video.write(new_im)
        video.release()


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


