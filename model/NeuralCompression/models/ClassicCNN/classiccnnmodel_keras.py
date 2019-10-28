import tensorflow as tf
import numpy as np
import model.NeuralCompression.models.ClassicCNN.compression as cpr
import os
import re
import binary_layer as bl


class BinCNNModel:

    def __init__(self, num_binary_filter_conv, num_binary_filter_dense):
        self.num_bin_filter_conv = num_binary_filter_conv
        self.num_bin_filter_dense = num_binary_filter_dense
        self.model = tf.keras.Sequential()

    def save_weights_for_FPGA(self, save_dir, alpha_fixp_format, bias_fixp_format):

        if not os.path.exists(save_dir):
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        bin_weight_file = open(save_dir + 'binweight.bwf', 'w', buffering=1)
        alpha_file = open(save_dir + 'alphaweight.awf', 'w', buffering=1)
        bias_file = open(save_dir + 'bias.baf', 'w', buffering=1)

        # write header
        bin_weight_file.writelines('# File containing the binary weights\n')
        bin_weight_file.writelines('# LAYER - TYPE: {SHAPE}\n')

        alpha_file.writelines('# File containing the alphas\n')
        alpha_file.writelines('# LAYER - TYPE: {SHAPE} {FORMAT}\n')

        bias_file.writelines('# File containing the bias\n')
        bias_file.writelines('# LAYER - TYPE: {SHAPE} {FORMAT}\n')

        with tf.compat.v1.keras.backend.get_session().as_default():
            for curr_layer_idx, curr_layer in enumerate(self.model.layers):
                if re.findall('binary_convolution', curr_layer.name):
                    curr_alpha_fixp_f = alpha_fixp_format.pop(0)
                    curr_bias_fixp_f = bias_fixp_format.pop(0)
                    kernel_width = curr_layer.trainable_variables[0].shape.as_list()[1]
                    kernel_height = curr_layer.trainable_variables[0].shape.as_list()[0]
                    kernel_depth = curr_layer.trainable_variables[0].shape.as_list()[2]
                    kernel_filters = curr_layer.trainable_variables[0].shape.as_list()[3]
                    bin_weight_file.writelines(
                        '{:d} - BCONV: {:d} {:d} {:d} {:d} {:d}\n'.format(curr_layer_idx, kernel_height, kernel_width,
                                                                          kernel_depth,
                                                                          kernel_filters,
                                                                          self.num_bin_filter_conv))
                    alpha_file.writelines(
                        '{:d} : {:d} {:d} W8_{:d}\n'.format(curr_layer_idx, kernel_filters, self.num_bin_filter_conv,
                                                            curr_alpha_fixp_f))
                    bias_file.writelines(
                        '{:d} : {:d} W8_{:d}\n'.format(curr_layer_idx, kernel_filters, curr_bias_fixp_f))
                    for curr_layer_var_idx, curr_layer_train_var in enumerate(curr_layer.trainable_variables):
                        re_bc = re.findall('binKernel_(\d)', curr_layer_train_var.name)
                        re_al = re.findall('alpha_(\d)', curr_layer_train_var.name)
                        re_bias = re.findall('bias', curr_layer_train_var.name)
                        if re_bc:
                            bin_kernel_array = np.sign(curr_layer_train_var.eval()).astype(np.int)
                            for channels in range(bin_kernel_array.shape[-1]):
                                with np.nditer(bin_kernel_array[:, :, :, channels], order='F') as it:
                                    while not it.finished:
                                        bin_weight_file.write(str(it.value) + ' ')
                                        it.iternext()
                                bin_weight_file.write('\n')
                            bin_weight_file.write('\n')

                        elif re_al:
                            alpha_array = (curr_layer_train_var.eval() * 2 ** curr_alpha_fixp_f).astype(np.int)
                            for alpha_m in np.nditer(alpha_array):
                                alpha_file.write(str(alpha_m) + ' ')
                            alpha_file.write('\n')

                        elif re_bias:
                            bias_array = (curr_layer_train_var.eval() * 2 ** curr_alpha_fixp_f).astype(np.int)
                            for bias_c in np.nditer(bias_array):
                                bias_file.write(str(bias_c) + ' ')
                            bias_file.write('\n')

                    bin_weight_file.writelines('\n')
                    alpha_file.writelines('\n')
                    bias_file.writelines('\n')

                elif re.match('binary_dense', curr_layer.name):

                    curr_alpha_fixp_f = alpha_fixp_format.pop(0)
                    curr_bias_fixp_f = bias_fixp_format.pop(0)
                    inputs = curr_layer.trainable_variables[0].shape.as_list()[0]
                    units = curr_layer.trainable_variables[0].shape.as_list()[1]
                    bin_weight_file.writelines(
                        '{:d} - BCONV: {:d} {:d} {:d}\n'.format(curr_layer_idx, inputs, units,
                                                                self.num_bin_filter_dense))
                    alpha_file.writelines(
                        '{:d} : {:d} {:d} W8_{:d}\n'.format(curr_layer_idx, units, self.num_bin_filter_conv,
                                                            curr_alpha_fixp_f))
                    bias_file.writelines('{:d} : {:d} W8_{:d}\n'.format(curr_layer_idx, units, curr_bias_fixp_f))

                    for curr_layer_var_idx, curr_layer_train_var in enumerate(curr_layer.trainable_variables):

                        re_bc = re.findall('binDenseKernel_(\d)', curr_layer_train_var.name)
                        re_al = re.findall('alpha_(\d)', curr_layer_train_var.name)
                        re_bias = re.findall('bias', curr_layer_train_var.name)
                        if re_bc:
                            bin_kernel_array = np.sign(curr_layer_train_var.eval()).astype(np.int)
                            for unit in range(bin_kernel_array.shape[-1]):
                                with np.nditer(bin_kernel_array[:, unit]) as it:
                                    while not it.finished:
                                        bin_weight_file.write(str(it.value) + ' ')
                                        it.iternext()
                                bin_weight_file.write('\n')
                                bin_weight_file.flush()
                            bin_weight_file.write('\n')

                        elif re_al:
                            alpha_array = (curr_layer_train_var.eval() * 2 ** curr_alpha_fixp_f).astype(np.int)
                            for alpha_m in np.nditer(alpha_array):
                                alpha_file.write(str(alpha_m) + ' ')
                            alpha_file.write('\n')

                        elif re_bias:
                            bias_array = (curr_layer_train_var.eval() * 2 ** curr_bias_fixp_f).astype(np.int)
                            for bias_d in np.nditer(bias_array):
                                bias_file.write(str(bias_d) + ' ')
                            bias_file.write('\n')

                    bin_weight_file.writelines('\n')
                    alpha_file.writelines('\n')
                    bias_file.writelines('\n')


class CNNModelKeras:

    def __init__(self):
        self.model = tf.keras.Sequential()

    def build_model_from_config(self, config, dataset):
        conv_size = config.get("model").get("conv_size")
        kernel_size = config.get("model").get("kernel_size")
        num_filter = config.get("model").get("num_filter")
        pool_size = config.get("model").get("pool_size")
        conv_stride = config.get("model").get("conv_stride")
        dense_depth = config.get("model").get("dense_depth")
        dense_neurons = config.get("model").get("dense_neurons")
        dropout_factor = config.get("model").get("dropout_factor")

        # set input shape for model
        self.model.add(tf.keras.layers.Input(dataset.input_shape))

        # build sequential model according to the config file
        for i in range(conv_size):
            self.model.add(
                tf.keras.layers.Conv2D(filters=num_filter[i], kernel_size=kernel_size[i], strides=conv_stride[i],
                                       activation='relu'))
            self.model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size[i]))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dropout(rate=dropout_factor[0]))
        for i in range(dense_depth):
            self.model.add(tf.keras.layers.Dense(units=dense_neurons[i], activation='relu'))
            self.model.add(tf.keras.layers.Dropout(rate=dropout_factor[i + 1]))

        self.model.add(tf.keras.layers.Dense(units=dataset.num_classes, activation='softmax'))

        self.model.summary()

    def quantize_weights(self, original_net_ckpt, num_binary_filter_conv,
                         num_binary_filter_dense=None, opt_runs=100, build_binarized_model=True):

        # Loads the weights
        self.model.load_weights(original_net_ckpt)

        if not num_binary_filter_dense:
            num_binary_filter_dense = num_binary_filter_conv

        if build_binarized_model:
            bin_model = BinCNNModel(num_binary_filter_conv=num_binary_filter_conv,
                                    num_binary_filter_dense=num_binary_filter_dense)
            first_layer = True
            #bin_model.model.add(tf.keras.layers.Input(self.model.layers[0].input.shape[1:]))

        with tf.compat.v1.keras.backend.get_session().as_default():
            for i_l, l in enumerate(self.model.layers):
                if l.trainable_variables:
                    for i_v, tr_var in enumerate(l.trainable_variables):
                        if re.findall('conv2d_\d/kernel', tr_var.name) or re.findall('conv2d/kernel', tr_var.name):
                            print('Quantizing Layer: ' + tr_var.name + '......', end='', flush=True)
                            conv_weights = tr_var.eval()
                            conv_weights_approx, _, binary_filter, alpha = cpr.m_fold_binaryconv_tensor_approx_whole(
                                conv_weights,
                                num_binary_filter_conv,
                                opt_runs=opt_runs)

                            tr_var.assign(conv_weights_approx)
                            print('DONE')

                        elif re.findall('dense_\d/kernel', tr_var.name) or re.findall('dense/kernel', tr_var.name):
                            print('Quantizing Layer ' + tr_var.name + '......', end='', flush=True)
                            dense_weights = tr_var.eval()
                            dense_weights_approx, _, binary_filter, alpha = cpr.m_fold_binarydense_tensor_approx_channel(
                                dense_weights,
                                num_binary_filter_dense,
                                opt_runs=opt_runs)

                            tr_var.assign(dense_weights_approx)
                            print('DONE')

                        elif re.findall('conv2d_\d/bias', tr_var.name) or re.findall('conv2d/bias', tr_var.name):
                            bias = tr_var.eval()

                        elif re.findall('dense_\d/bias', tr_var.name) or re.findall('dense/bias', tr_var.name):
                            bias = tr_var.eval()

                    if build_binarized_model:
                        if re.findall('conv2d', l.name):
                            if first_layer:
                                bin_model.model.add(
                                    bl.BinaryConvolution(**l.get_config(), binary_filters=num_binary_filter_conv,
                                                         input_shape=self.model.layers[0].input.shape[1:]))
                                first_layer = False
                            else:
                                bin_model.model.add(
                                    bl.BinaryConvolution(**l.get_config(), binary_filters=num_binary_filter_conv))
                            for i_nl, nl_var in enumerate(bin_model.model.layers[-1].trainable_variables):

                                re_bc = re.findall('binKernel_(\d)', nl_var.name)
                                re_al = re.findall('alpha_(\d)', nl_var.name)
                                re_bias = re.findall('bias', nl_var.name)
                                if re_bc:
                                    idx_bc = int(re_bc[0])
                                    nl_var.assign(binary_filter[:, :, :, :, idx_bc]).eval()

                                elif re_al:
                                    idx_al = int(re_al[0])
                                    nl_var.assign(alpha[:, idx_al]).eval()

                                elif re_bias:
                                    nl_var.assign(bias).eval()

                        if re.findall('dense', l.name):
                            if first_layer:
                                bin_model.model.add(
                                    bl.BinaryDense(**l.get_config(), binary_units=num_binary_filter_dense,
                                                   input_shape=self.model.layers[0].input.shape[1:]))
                                first_layer = False
                            else:
                                bin_model.model.add(
                                    bl.BinaryDense(**l.get_config(), binary_units=num_binary_filter_dense))

                            for i_nl, nl_var in enumerate(bin_model.model.layers[-1].trainable_variables):
                                re_bc = re.findall('binDenseKernel_(\d)', nl_var.name)
                                re_al = re.findall('alpha_(\d)', nl_var.name)
                                re_bias = re.findall('bias', nl_var.name)
                                if re_bc:
                                    idx_bc = int(re_bc[0])
                                    nl_var.assign(binary_filter[:, :, idx_bc]).eval()

                                elif re_al:
                                    idx_al = int(re_al[0])
                                    nl_var.assign(alpha[:, idx_al]).eval()

                                elif re_bias:
                                    nl_var.assign(bias).eval()

                elif build_binarized_model:
                    bin_model.model.add(l)

        if build_binarized_model:
            return bin_model
