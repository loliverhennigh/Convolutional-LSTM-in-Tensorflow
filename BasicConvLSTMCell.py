
import tensorflow as tf

class RNNCell(object):
  """Abstract object representing an RNN cell.
  The definition of cell in this package differs from the definition used in the
  literature. In the literature, cell refers to an object with a single scalar
  output. The definition in this package refers to a horizontal array of such
  units.
  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  tuple of integers, then it results in a tuple of `len(state_size)` state
  matrices, each with the a column size corresponding to values in `state_size`.
  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by the class `MultiRNNCell`,
  or by calling the `rnn` ops several times. Every `RNNCell` must have the
  properties below and and implement `__call__` with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.
    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = self.state_size
    if nest.is_sequence(state_size):
      state_size_flat = nest.flatten(state_size)
      zeros_flat = [
          array_ops.zeros(
              array_ops.pack(_state_size_with_prefix(s, prefix=[batch_size])),
              dtype=dtype)
          for s in state_size_flat]
      for s, z in zip(state_size_flat, zeros_flat):
        z.set_shape(_state_size_with_prefix(s, prefix=[None]))
      zeros = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=zeros_flat)
    else:
      zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
      zeros = array_ops.zeros(array_ops.pack(zeros_size), dtype=dtype)
      zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

    return zeros


class BasicConvLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
               state_is_tuple=False, activation=tf.nn.tanh):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    #if not state_is_tuple:
      #logging.warn("%s: Using a concatenated state is slower and will soon be "
      #             "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self.shape = shape 
    self.filter_size = filter_size
    self.num_features = num_features 
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(3, 2, state)
      concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(3, 4, concat)

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat(3, [new_c, new_h])
      return new_h, new_state

def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Conv"):
    matrix = tf.get_variable(
        "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(tf.concat(3, args), matrix, strides=[1, 1, 1, 1], padding='SAME')
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [num_features],
        dtype=dtype,
        initializer=tf.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term



