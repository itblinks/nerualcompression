import tensorflow as tf
import os, sys, glob
import re
import ast
from models.ClassicCNN import compression


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size,
    in contrast to Tensorflows conv2d padding scheme, which pads according to
    the input size.
    Example with as stride of 2
    Tensorflow Conv2d:
        case 1:
                    pad|              |pad
        inputs:      0 |1  2  3  4  5 |0
                       |_______|
                             |_______|
                                   |_______|

        case 2:
                                      |pad
        inputs:      1  2  3  4  5  6 |0
                    |_______|
                         |_______|
                              |_______|

    This implementation
       case 1:
                   pad|              |pad
       inputs:      0 |1  2  3  4  5 |0
                   |_______|
                         |_______|
                               |_______|

        case 2:
                    pad|                 |pad
        inputs:      0 |1  2  3  4  5  6 |0
                    |_______|
                          |_______|
                                |_______|

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def dense_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights)
        with tf.name_scope('biases'):
            bias = tf.Variable(tf.truncated_normal([output_dim], stddev=1))
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + bias
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations


def conv_layer(inputs, filters, kernel_size, strides, data_format, layer_name, padding, act=tf.nn.relu):
    """Reusable code for making a convolution neural net layer.

    It does a 2d matrix convolution with separate bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, data_format)
        with tf.name_scope('ConvWeights'):
            conv = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=act,
                use_bias=True)
            c2_vars = tf.trainable_variables()
            tf.summary.histogram("weights", c2_vars[0])
            tf.summary.histogram("biases", c2_vars[1])
            tf.summary.histogram("activations", conv)
        return conv


class Model(object):

    def __init__(self, conv_size, kernel_size, num_filter, pool_size, pool_stride, conv_stride,
                 data_format, dense_depth, dense_neurons, dropout, num_classes, quant_act_format, ID, parseID,
                 arch_file):
        self.__conv_size = conv_size
        self.__kernel_size = kernel_size
        self.__num_filter = num_filter
        self.__dense_depth = dense_depth
        self.__dense_neurons = dense_neurons
        self.__dropout = dropout
        self.__pool_size = pool_size
        self.__pool_stride = pool_stride
        self.__conv_stride = conv_stride
        self.__data_format = data_format
        self.__num_classes = num_classes
        self.__quant_act_format = quant_act_format
        self.__ID = ID
        self.__parseID = parseID
        self.__arch_file = arch_file

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.
        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.
        Returns:
          A variable scope for the model.
        """

        return tf.variable_scope('ClassicCNN',
                                 custom_getter=self._custom_dtype_getter)

    def parseNetworkArchitecture(self, inputs, training, quant_act=False,
                                 quant_act_format=None):
        if self.__parseID:
            # parse the arch file to get a list with all needed layers
            raw_arch_file = open(self.__arch_file)
            for i, line in enumerate(raw_arch_file):
                if i == self.__ID:
                    rawID = line

            raw_splitted = rawID.split('___')
            layer_list = list()
            for i, raw_layer in enumerate(raw_splitted):
                if i != 0:  # ignore first line, this only contains the ID
                    # separate at the first { where the type ends. Add a {, since it is removed by the split method
                    layer_list.append(ast.literal_eval('{' + raw_layer.split('{')[1]))
                    layer_list[i - 1]['Type'] = raw_layer.split('{')[0]  # Add the type back to the dictionary

            count_layer_types = {'C': 0, 'D': 0, 'Drop': 0, 'F': 0}
            act_nr = 0

            # parse the input
            for i, layer in enumerate(layer_list):

                # check if layer is conv type
                if layer['Type'] == 'C':
                    with tf.name_scope('Convolution/'):
                        with tf.variable_scope('conv2d_{}'.format(count_layer_types['C'] + 1)):
                            inputs = conv_layer(inputs=inputs, filters=layer['nbr_of_filters'],
                                                kernel_size=layer['kernel_size'],
                                                strides=layer['stride'], data_format=tf.float32,
                                                layer_name='conv2d_{}'.format(count_layer_types['C'] + 1),
                                                padding=layer['padding'],
                                                act=tf.nn.relu)
                            count_layer_types['C'] += 1

                            if quant_act:
                                inputs = tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                                      quant_act_format[act_nr][0],
                                                                                      quant_act_format[act_nr][1],
                                                                                      quant_act_format[act_nr][2])
                                act_nr += 1

                # check if layer is pool type
                elif layer['Type'] == 'P':
                    with tf.name_scope('Convolution/'):
                        with tf.variable_scope('conv2d_{}'.format(count_layer_types['C'])):
                            inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=layer['pool_size'],
                                                             strides=layer['stride'], padding=layer['padding'])

                # check if layer is dropout type
                elif layer['Type'] == 'Drop':
                    with tf.name_scope('Convolution/'):
                        inputs = tf.layers.dropout(inputs, rate=layer['dropout_rate'],
                                                   training=training == tf.estimator.ModeKeys.TRAIN)

                # check if layer is dense type
                elif layer['Type'] == 'D':
                    # check if it is the last dense layer e.g the logits layer
                    with tf.name_scope('Dense/'):
                        if i != len(layer_list) - 1:
                            with tf.variable_scope('dense_{}'.format(count_layer_types['D'] + 1)):
                                inputs = dense_layer(input_tensor=inputs, input_dim=inputs.get_shape().as_list()[1],
                                                     output_dim=layer['nbr_of_neurons'],
                                                     layer_name='dense_{}'.format(count_layer_types['D'] + 1),
                                                     act=tf.nn.relu)
                                count_layer_types['D'] += 1

                                if layer['hasDropOut']:
                                    inputs = tf.layers.dropout(inputs, rate=self.__dropout[0],
                                                               training=training == tf.estimator.ModeKeys.TRAIN)
                        else:
                            inputs = dense_layer(input_tensor=inputs, input_dim=inputs.get_shape().as_list()[1],
                                                 output_dim=self.__num_classes,
                                                 layer_name='logits', act=tf.identity)

                # check if layer is flatten
                elif layer['Type'] == 'F':
                    with tf.name_scope('Dense'):
                        inputs = tf.layers.flatten(inputs)
                else:
                    raise NotImplementedError('This layer is not implementet in the layer list: ' + layer['Type'])
            self.__conv_size = count_layer_types['C']
            self.__dense_depth = count_layer_types['D']
        else:
            if quant_act:
                act_nr = 0
                inputs = tf.quantization.fake_quant_with_min_max_vars(inputs, quant_act_format[act_nr][0],
                                                                      quant_act_format[act_nr][1],
                                                                      quant_act_format[act_nr][2])
                act_nr += 1
            with tf.name_scope('Convolution'):
                for i, curr_kernel_size in enumerate(self.__kernel_size):
                    with tf.variable_scope('conv2d_{}'.format(i + 1)):
                        inputs = conv_layer(inputs=inputs, filters=self.__num_filter[i], kernel_size=curr_kernel_size,
                                            strides=self.__conv_stride[i], data_format=self.__data_format,
                                            layer_name='conv2d'.format(i + 1), act=tf.nn.relu, padding='valid')
                        inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=self.__pool_stride[:][i],
                                                         strides=self.__pool_stride[:][i])
                        if quant_act:
                            inputs = tf.quantization.fake_quant_with_min_max_vars(inputs, quant_act_format[act_nr][0],
                                                                                  quant_act_format[act_nr][1],
                                                                                  quant_act_format[act_nr][2])
                            act_nr += 1

            with tf.name_scope('Dense'):
                inputs = tf.layers.flatten(inputs)

                inputs = tf.layers.dropout(inputs, rate=self.__dropout[0],
                                           training=training == tf.estimator.ModeKeys.TRAIN)

                for i in range(self.__dense_depth):
                    with tf.variable_scope('dense_{}'.format(i + 1)):
                        inputs = dense_layer(input_tensor=inputs, input_dim=inputs.get_shape().as_list()[1],
                                             output_dim=self.__dense_neurons[i], layer_name='dense_{}'.format(i + 1),
                                             act=tf.nn.relu)
                        inputs = tf.layers.dropout(inputs, rate=self.__dropout[i + 1],
                                                   training=training == tf.estimator.ModeKeys.TRAIN)
                        if quant_act:
                            inputs = tf.quantization.fake_quant_with_min_max_vars(inputs, quant_act_format[act_nr][0],
                                                                                  quant_act_format[act_nr][1],
                                                                                  quant_act_format[act_nr][2])
                            act_nr += 1

                inputs = dense_layer(input_tensor=inputs, input_dim=inputs.get_shape().as_list()[1],
                                     output_dim=self.__num_classes,
                                     layer_name='logits', act=tf.identity)
                if quant_act:
                    inputs = tf.quantization.fake_quant_with_min_max_vars(inputs, quant_act_format[act_nr][0],
                                                                          quant_act_format[act_nr][1],
                                                                          quant_act_format[act_nr][2])
                    act_nr += 1
        return inputs

    def classiccnn_model_fn(self, features, labels, mode, params):
        # This acts as a no-op if the logits are already in fp32 (provided logits are
        # not a SparseTensor). If dtype is is low precision, logits must be cast to
        # fp32 for numerical stability.
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        net = tf.reshape(net, shape=[-1] + list(params['feature_columns'].shape))

        logits = self.parseNetworkArchitecture(inputs=net, training=mode, quant_act=params['quant_act'],
                                               quant_act_format=self.__quant_act_format)
        logits = tf.cast(logits, tf.float32)

        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
            'image': net
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Return the predictions and the specification for serving a SavedModel
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        # Calculate loss, which includes softmax cross entropy
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        loss = cross_entropy

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()

            if params['decay']:
                learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step,
                                                           params['decay_steps'], params['decay_rate'], staircase=True)
            else:
                learning_rate = tf.constant(params['learning_rate'])

            # Create a tensor named learning_rate for logging purposes
            tf.identity(learning_rate, name='learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
            )

            train_op = optimizer.minimize(cross_entropy, global_step=global_step)

        else:
            train_op = None
            global_step = 0
            learning_rate = 0

        accuracy = tf.metrics.accuracy(labels, predictions['classes'])

        metrics = {'accuracy': accuracy}

        logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                   "accuracy": accuracy[1],
                                                   "step": global_step,
                                                   "learning_rate": learning_rate}
                                                  , every_n_iter=500)

        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_hooks=[logging_hook]
        )

    def quantize_weights(self, model_dir, quant_param, debug_plots=False):
        def extract_step_nr(file):
            s = re.findall("\d+", file)[-1]
            return (int(s) if s else -1, file)

        list_of_graphs = glob.glob('{}/*.meta'.format(model_dir))
        last_graph = max(list_of_graphs, key=extract_step_nr)
        saver = tf.train.import_meta_graph(last_graph)
        with tf.Session() as sess:
            saver.restore(sess, os.path.splitext(last_graph)[0])

            if quant_param['quant_conv'] is True:
                for i in range(self.__conv_size):
                    conv_kernel_tensor = tf.trainable_variables(
                        scope='conv2d_{}'.format(i + 1))  # first tensor are kernel weights in this model
                    conv_weights = conv_kernel_tensor[0].eval()
                    conv_weights_approx = conv_weights
                    if quant_param['bin_quant'] is True:
                        conv_weights_approx, _ = compression.m_fold_binaryconv_tensor_approx_whole(conv_weights,
                                                                                                   num_binary_filter=
                                                                                                   quant_param[
                                                                                                       'num_bin_filt_conv'],
                                                                                                   paper_approach=
                                                                                                   quant_param[
                                                                                                       'paper_approach'],
                                                                                                   use_pow_two=
                                                                                                   quant_param[
                                                                                                       'pow_of_two'])
                    if quant_param['log_quant'] is True:
                        conv_weights_approx, _ = compression.log_approx(conv_weights, result_plots=False)
                    conv_kernel_tensor[0].load(conv_weights_approx, sess)
            if quant_param['quant_dense'] is True:
                for i in range(self.__dense_depth):
                    dense_weights_tensor = tf.trainable_variables(
                        scope='Dense/dense_{}'.format(i + 1))  # first tensor are kernel weights in this model
                    dense_weights = dense_weights_tensor[0].eval()
                    dense_weights_approx = dense_weights
                    if quant_param['bin_quant'] is True:
                        dense_weights_approx, _ = compression.m_fold_binarydense_tensor_approx_channel(dense_weights,
                                                                                                       num_binary_filter=
                                                                                                       quant_param[
                                                                                                           'num_bin_filt_dense'],
                                                                                                       paper_approach=
                                                                                                       quant_param[
                                                                                                           'paper_approach'],
                                                                                                       use_pow_two=
                                                                                                       quant_param[
                                                                                                           'pow_of_two'])
                    if quant_param['log_quant'] is True:
                        dense_weights_approx, _ = compression.log_approx(dense_weights, result_plots=False)
                    dense_weights_tensor[0].load(dense_weights_approx, sess)
                logits_weights_tensor = tf.trainable_variables(
                    scope='Dense/logits'.format(i + 1))  # first tensor are kernel weights in this model
                logits_weights = logits_weights_tensor[0].eval()
                logits_weights_approx = logits_weights
                if quant_param['bin_quant'] is True:
                    logits_weights_approx, _ = compression.m_fold_binarydense_tensor_approx_channel(logits_weights,
                                                                                                    num_binary_filter=
                                                                                                    quant_param[
                                                                                                        'num_bin_filt_dense'],
                                                                                                    paper_approach=
                                                                                                    quant_param[
                                                                                                        'paper_approach'],
                                                                                                    use_pow_two=
                                                                                                    quant_param[
                                                                                                        'pow_of_two'])
                if quant_param['log_quant'] is True:
                    logits_weights_approx, _ = compression.log_approx(logits_weights)

                logits_weights_tensor[0].load(logits_weights_approx, sess)

            if not os.path.exists(model_dir + '_quantized'):
                os.mkdir(model_dir + '_quantized')
            saver.save(sess, model_dir + '_quantized/model_quant.ckpt', write_meta_graph=False)
