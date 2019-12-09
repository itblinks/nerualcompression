import yaml
import numpy as np
import os
import re
import tensorflow as tf
import hashlib
import models.ClassicCNN.classiccnnmodel_keras as cnn
import binary_layer as bl
import dataset as ds
import datetime
import matplotlib.pyplot as plt

batch_size = 32
num_epochs = 30
train = False
retrain = False
num_epochs_retrain = 10
quant = False
eval_quant = True
k = 1
num_bin_filter_conv = 2
num_bin_filter_dense = 2


def main():
    # import yaml config file to set up network
    config_file = open("config.yaml")

    config = yaml.load(config_file)
    dataset = ds.Dataset(config.get("dataset"), use_data_api=False)

    # build model from config and dataset
    net = cnn.CNNModelKeras()
    net.build_model_from_config(config, dataset)

    # print a hash of the configuration
    hashfun = hashlib.sha1()
    hashfun.update(str(config.get("model")).encode('utf-8'))
    hashfun.update(str(config.get("dataset")).encode('utf-8'))
    digest = hashfun.hexdigest()
    print('The model-config has hash: {}'.format(digest))

    if train:
        net_dir = "trained_models/" + '{:s}_'.format(digest) + datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        os.makedirs(net_dir, exist_ok=True)
        os.makedirs(net_dir + '/real_weights', exist_ok=True)
        os.makedirs(net_dir + '/logs/real', exist_ok=True)
        net.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Define the Keras TensorBoard callback.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=net_dir + '/logs/real')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=net_dir + '/real_weights/real_model.ckpt',
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         verbose=1)

        net.model.fit(dataset.train[0]['image'], dataset.train[1], epochs=num_epochs,
                      validation_data=(dataset.test[0]['image'], dataset.test[1]),
                      callbacks=[tensorboard_callback, cp_callback])
        net.model.load_weights(net_dir + '/real_weights/real_model.ckpt')
        score = net.model.evaluate(dataset.test[0]['image'], dataset.test[1], callbacks=[tensorboard_callback])
        print('Final Accuracy on validation before quantization: {:.2f}'.format(
            score[-1] * 100))

    if quant:
        print('Listing available checkpoints of the specified network')
        valid_dirs = [val_dir for val_dir in os.listdir('trained_models') if re.match(digest, val_dir)]
        for idx, directory in enumerate(valid_dirs):
            print('{:d}: {:s}'.format(idx, directory))
        net_dir = 'trained_models/' + valid_dirs[int(input('Select a dictionary: '))]
        original_net_ckpt = net_dir + '/real_weights/real_model.ckpt'
        os.makedirs(net_dir + '/bin_weights', exist_ok=True)
        bin_net_ckpt = net_dir + '/bin_weights'

        os.makedirs(net_dir + '/logs/bin', exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=net_dir + '/logs/bin')

        bin_net = net.quantize_weights(original_net_ckpt, num_bin_filter_conv, num_bin_filter_dense,
                                       opt_runs=100, build_binarized_model=True)

        net.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        net.model.evaluate(dataset.test[0]['image'], dataset.test[1], callbacks=[tensorboard_callback])

        bin_net.model.summary()
        bin_net.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        score = bin_net.model.evaluate(dataset.test[0]['image'], dataset.test[1], callbacks=[tensorboard_callback])
        print('Final Accuracy on validation after quantization before retraining: {:.2f}'.format(
            score[-1] * 100))

        bin_net.model.save(bin_net_ckpt + '/bin_model.h5')
        os.makedirs(bin_net_ckpt + '/folded', exist_ok=True)
        net.model.save_weights(bin_net_ckpt + '/folded/folded_model.ckpt')

    if retrain:
        try:
            bin_net
        except NameError:
            print('Net needs to be built first!')
            print('Listing available checkpoints of the specified network')
            valid_dirs = [val_dir for val_dir in os.listdir('trained_models') if re.match(digest, val_dir)]
            for idx, directory in enumerate(valid_dirs):
                print('{:d}: {:s}'.format(idx, directory))
            bin_net_ckpt = 'trained_models/' + valid_dirs[int(input('Select a dictionary: '))] + '/bin_weights'

            bin_net = cnn.BinCNNModel(num_binary_filter_conv=num_bin_filter_conv,
                                      num_binary_filter_dense=num_bin_filter_dense)

        bin_net.model = tf.keras.models.load_model(bin_net_ckpt + '/bin_model.h5',
                                                   custom_objects={'BinaryDense': bl.BinaryDense,
                                                                   'BinaryConvolution': bl.BinaryConvolution})

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=bin_net_ckpt + '/bin_model.h5',
                                                         save_best_only=True,
                                                         save_weights_only=False,
                                                         verbose=1)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=bin_net_ckpt + '/../' + '/logs/bin')
        bin_net.model.fit(dataset.train[0]['image'], dataset.train[1], epochs=num_epochs_retrain,
                          validation_data=(dataset.test[0]['image'], dataset.test[1]),
                          callbacks=[tensorboard_callback, cp_callback])

    if eval_quant:
        try:
            bin_net
        except NameError:
            print('Net needs to be built first!')
            print('Listing available checkpoints of the specified network')
            valid_dirs = [val_dir for val_dir in os.listdir('trained_models') if re.match(digest, val_dir)]
            for idx, directory in enumerate(valid_dirs):
                print('{:d}: {:s}'.format(idx, directory))
            bin_net_ckpt = 'trained_models/' + valid_dirs[int(input('Select a dictionary: '))] + '/bin_weights'

            bin_net = cnn.BinCNNModel(num_binary_filter_conv=num_bin_filter_conv,
                                      num_binary_filter_dense=num_bin_filter_dense)

            bin_net.model = tf.keras.models.load_model(bin_net_ckpt + '/bin_model.h5',
                                                       custom_objects={'BinaryDense': bl.BinaryDense,
                                                                       'BinaryConvolution': bl.BinaryConvolution})

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=bin_net_ckpt + '/../' + '/logs/bin')
        score = bin_net.model.evaluate(dataset.test[0]['image'], dataset.test[1],
                                       callbacks=[tensorboard_callback])
        print(
            'Final Accuracy on validation after quantization after retraining: {:.2f}'.format(score[-1] * 100))

        bin_net.save_weights_for_FPGA(bin_net_ckpt + '/weights_output/', [11, 9, 9, 9, 9], [8, 7, 7, 7, 7])


if __name__ == '__main__':
    main()
