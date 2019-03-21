import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 




class Downsample(tf.keras.Model):
    def __init__(self,filters,kernel_size,apply_batchnorm=True,is_training=True):
        super(Downsample, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size,
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                                use_bias=True,
                                bias_initializer=keras.initializers.Constant(value=0.0))
        self.use_bn=apply_batchnorm
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                        momentum=0.9,
                                                        scale=True,
                                                        trainable=is_training)
        self.activation=layers.LeakyReLU(alpha=0.2)

    def call(self,x,training):
        x=self.conv(x)
        if self.use_bn:
            x=self.bn(x,training=training)
        x=self.activation(x)
        return x           

class DenseLayer(tf.keras.Model):
    def __init__(self,hidden_n,activation=None,apply_batchnorm=True,input_dim=None,is_training=True):
        super(DenseLayer, self).__init__()
        self.activation=activation
        self.use_bn=apply_batchnorm
        if self.use_bn:

            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                        momentum=0.9,
                                                        scale=True,
                                                        trainable=is_training)
        if not input_dim:
            self.fc = layers.Dense(hidden_n, input_shape=(input_dim,),
                                    kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                    bias_initializer=keras.initializers.Constant(value=0.0))

        else:
            self.fc = layers.Dense(hidden_n, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                    bias_initializer=keras.initializers.Constant(value=0.0))

        if self.activation=='relu':
            self.activation=tf.keras.layers.ReLU()
        elif self.activation=='lrelu':
            self.activation=tf.keras.layers.LeakyReLU(alpha=0.2)


    def call(self,x,training):
        x=self.fc(x)
        if self.use_bn:
            x=self.bn(x,training=training)
        if self.activation:
            x=self.activation(x)
        return x



class Upsample(tf.keras.Model):
  def __init__(self, filters, kernel_size,apply_batchnorm=True,is_training=True):
    super(Upsample, self).__init__()
    self.use_bn=apply_batchnorm
    self.up_conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                                   use_bias=True,
                                                   bias_initializer=keras.initializers.Constant(value=0.0))
    if self.use_bn:
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                            momentum=0.9,
                                            scale=True,
                                            trainable=is_training)
        self.activation=layers.ReLU()
        
  def call(self, x,training):
    x = self.up_conv(x)
    if self.use_bn:  
        x=self.bn(x,training=training)
        x=self.activation(x)
    else :
        x = keras.activations.sigmoid(x)       
    return x


def conv_cond_concat(x, y):
    
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    # print(x_shapes)
    y_shapes = tf.shape(y)
    y=tf.reshape(y,[-1,1,1,y_shapes[1]])
    y_shapes=tf.shape(y)

    # print(y_shapes)
    
    return tf.concat([x, y*tf.ones([x_shapes[0],x_shapes[1], x_shapes[2], y_shapes[3]])], 3)