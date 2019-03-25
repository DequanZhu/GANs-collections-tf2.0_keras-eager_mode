import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




class Sigmoid(layers.Layer):

  def __init__(self):
    super(Sigmoid, self).__init__()

  def call(self, inputs):
    return keras.activations.sigmoid(inputs)
    

    

class Tanh(layers.Layer):
  def __init__(self):
    super(Tanh, self).__init__()

  def call(self, inputs):
    return keras.activations.tanh(inputs)




def Conv2D(filters, kernel_size, strides, activation=None):

    conv_op = layers.Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            activation=activation,
                            padding='same',
                            kernel_initializer=keras.initializers.TruncatedNormal(
                                stddev=0.02),
                            use_bias=True,
                            bias_initializer=keras.initializers.Constant(value=0.0))
    return conv_op


def BatchNorm(is_training=True):
    bn_op = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                               momentum=0.9,
                                               scale=True,
                                               trainable=is_training)
    return bn_op


def DenseLayer(hidden_n, is_input=False, input_dim=None, activation=None):
    if activation == 'lrelu':
        activation = layers.LeakyReLU(alpha=0.2)
    if is_input:
        fc_op = layers.Dense(hidden_n, input_shape=(input_dim,),
                             activations=activation,
                             kernel_initializer=keras.initializers.RandomNormal(
                                 stddev=0.02),
                             bias_initializer=keras.initializers.Constant(value=0.0))
    else:
        fc_op = layers.Dense(hidden_n,
                             kernel_initializer=keras.initializers.RandomNormal(
                                 stddev=0.02),
                             bias_initializer=keras.initializers.Constant(value=0.0))
    return fc_op


def UpConv2D(filters, kernel_size, strides, activation=None):
    up_conv_op = layers.Conv2DTranspose(filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        activation=activation,
                                        kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                        use_bias=True,
                                        bias_initializer=keras.initializers.Constant(value=0.0))
    return up_conv_op


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = tf.shape(y)
    y = tf.reshape(y, [-1, 1, 1, y_shapes[1]])
    y_shapes = tf.shape(y)
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
