import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class BinaryDense(layers.Layer):
    def __init__(self, units, binary_units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super(BinaryDense, self).__init__(**kwargs)
        self.binary_units = binary_units
        self.units = units
        self.bias_constraint = bias_constraint
        self.kernel_constraint = kernel_constraint
        self.activity_regularizer = activity_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.activation = activation

    def get_config(self):
        config = {
            'binary_units': self.binary_units,
            'units': self.units,
            'bias_constraint': self.bias_constraint,
            'kernel_constraint': self.kernel_constraint,
            'activity_regularizer': self.activity_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_initializer': self.bias_initializer,
            'kernel_initializer': self.kernel_initializer,
            'use_bias': self.use_bias,
            'activation': self.activation,
        }
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):

        self.b = []
        self.a = []
        for b_i in range(self.binary_units):
            self.b.append(self.add_weight(shape=([input_shape[-1]] + [self.units]),
                                          initializer='ones',
                                          trainable=True, name='binDenseKernel_' + str(b_i)))
            self.a.append(self.add_weight(shape=[self.units], initializer='ones', name='alpha_' + str(b_i)))

        self.bias = self.add_weight(shape=[self.units], name='bias')
        super(BinaryDense, self).build(input_shape)

    def call(self, inputs):

        @tf.custom_gradient
        def sign_straight_through(x):
            def grad(dy):
                return dy * tf.where(tf.math.greater_equal(x, -tf.ones_like(x)), tf.ones_like(dy),
                                     tf.zeros_like(dy)) * tf.where(
                    tf.math.less_equal(x, tf.ones_like(x)), tf.ones_like(dy), tf.zeros_like(dy))

            return tf.sign(x), grad

        out = self.bias
        for b_i in range(self.binary_units):
            b = sign_straight_through(self.b[b_i])
            dense = tf.matmul(inputs, b)

            dense = dense * self.a[b_i]

            out = tf.add(dense, out)
        return tf.keras.layers.Activation(self.activation)(out)


class BinaryConvolution(layers.Layer):

    def __init__(self, filters, kernel_size, binary_filters, strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(BinaryConvolution, self).__init__(**kwargs)
        self.strides = strides
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.binary_filters = binary_filters
        self.padding = padding
        self.activation = activation
        self.activity_regularizer = activity_regularizer
        self.filters = filters
        self.kernel_size = kernel_size

    def get_config(self):

        config = {
            'strides': self.strides,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'use_bias': self.use_bias,
            'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'binary_filters': self.binary_filters,
            'padding': self.padding,
            'activation': self.activation,
            'activity_regularizer': self.activity_regularizer,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        }
        base_config = super(BinaryConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):

        self.b = []
        self.a = []
        for b_i in range(self.binary_filters):
            self.b.append(self.add_weight(shape=(list(self.kernel_size) + [int(input_shape[-1])] + [self.filters]),
                                          initializer='ones',
                                          trainable=True, name='binKernel_' + str(b_i)))
            self.a.append(self.add_weight(shape=[self.filters], initializer='ones', name='alpha_' + str(b_i)))

        self.bias = self.add_weight(shape=[self.filters], name='bias')
        super(BinaryConvolution, self).build(input_shape)

    def call(self, inputs):

        out = self.bias
        for b_i in range(self.binary_filters):
            @tf.custom_gradient
            def sign_straight_through(x):
                def grad(dy):
                    return dy * tf.where(tf.math.greater_equal(x, -tf.ones_like(x)), tf.ones_like(dy),
                                         tf.zeros_like(dy)) * tf.where(
                        tf.math.less_equal(x, tf.ones_like(x)), tf.ones_like(dy), tf.zeros_like(dy))

                return tf.sign(x), grad

            b = sign_straight_through(self.b[b_i])
            conv = tf.compat.v1.nn.conv2d(inputs, b, strides=self.strides, padding=self.padding.upper())
            conv = conv * self.a[b_i]

            out = tf.add(conv, out)
        return tf.keras.layers.Activation(self.activation)(out)


if __name__ == '__main__':
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((48, 48, 3)))
    model.add(BinaryConvolution(8, (3, 3), 2))
    model.add(BinaryConvolution(16, (3, 3), 2))
    model.summary()
