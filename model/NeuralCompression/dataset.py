import tensorflow as tf
import numpy as np
from backend.CNN_ImageDataHandler import CNN_ImageDataHandler
from skimage import exposure
import h5py
import os, sys, pathlib
import requests


def loadGTSRB48x48(raw=False):
    try:
        os.scandir('../../dataset/GTSRB48x48')
        f = open('../../dataset/GTSRB48x48/dataset_GTSRB_original_48_train.h5')
        f.close()
        f = open('../../dataset/GTSRB48x48/dataset_GTSRB_original_48_test.h5')
        f.close()
    except FileNotFoundError:
        pathlib.Path('../../dataset/GTSRB48x48').mkdir(parents=True, exist_ok=True)
        GTSRB48x48_files = {
            'dataset_GTSRB_original_48_train.h5': 'https://drive.switch.ch/index.php/s/i6yH9BzFu3UYbi5/download',
            'dataset_GTSRB_original_48_test.h5': 'https://drive.switch.ch/index.php/s/xjwc8IeHBBBGYTu/download'}
        for file_name, link in GTSRB48x48_files.items():
            with open('../../dataset/GTSRB48x48/' + file_name, "wb") as f:
                print("Downloading %s" % file_name)
                response = requests.get(link, stream=True)
                total_length = response.headers.get('content-length')

                if total_length is None:  # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s%s]" % ('=' * done, '>', ' ' * (50 - done)) + "%3.2f" % (100*dl/total_length) + "%")
                        sys.stdout.flush()
                    sys.stdout.write("\n")
                    f.close()
    training_file = '../../dataset/GTSRB48x48/dataset_GTSRB_original_48_train.h5'
    testing_file = '../../dataset/GTSRB48x48/dataset_GTSRB_original_48_test.h5'
    # Load the training data from the h5 files
    h5_file_train = h5py.File(training_file, 'r')
    x_train = np.array(h5_file_train.get('input'), dtype=np.float32)
    y_train = np.array(h5_file_train.get('labels'), dtype=np.int8)
    h5_file_train.close()

    # Load the test data from the h5 files
    h5_file_test = h5py.File(testing_file, 'r')
    x_test = np.array(h5_file_test.get('input'), dtype=np.float32)
    y_test = np.array(h5_file_test.get('labels'), dtype=np.int8)
    h5_file_test.close()

    def image_norm(image):
        image_min = np.min(image, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        image_max = np.max(image, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        image_new = (image - image_min) * 1 / (image_max - image_min)
        return image_new

    def hist_equ(image):
        image_new = exposure.equalize_hist(image)
        return image_new

    x_train_norm = image_norm(x_train)
    x_train_norm_eq = hist_equ(x_train_norm)

    x_test_norm = image_norm(x_test)
    x_test_norm_eq = hist_equ(x_test_norm)
    if raw:
        test = (dict(image=np.reshape(np.asarray(x_test_norm_eq, dtype=np.float32), newshape=[-1] + [48, 48, 3])),
                np.asarray(y_test, dtype=np.int32))
        train = (dict(image=np.reshape(np.asarray(x_train_norm_eq, dtype=np.float32), newshape=[-1] + [48, 48, 3])),
                 np.asarray(y_train, dtype=np.int32))
    else:
        test, train = as_image_dataset(x_test_norm_eq, x_train_norm_eq, y_test, y_train, image_shape=[48, 48, 3])
    data_format = 'channels_last'
    num_classes = 43
    return train, test, data_format, num_classes


# define datasets
def loadMNIST(raw=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if raw:
        test = (dict(image=np.reshape(np.asarray(x_train, dtype=np.float32), newshape=[-1] + [28, 28, 1])),
                np.asarray(y_train, dtype=np.int32))
        train = (dict(image=np.reshape(np.asarray(x_test, dtype=np.float32), newshape=[-1] + [28, 28, 1])),
                 np.asarray(y_test, dtype=np.int32))
    else:
        test, train = as_image_dataset(x_test, x_train, y_test, y_train, image_shape=[28, 28, 1])
    data_format = 'channels_last'
    num_classes = 10
    return train, test, data_format, num_classes


def loadWheels(raw=True):
    dataPath = '../../dataset/Wheels_negative_extended'
    data = CNN_ImageDataHandler(dataPath, one_hot=False, validation_size=200)
    # get data
    x_train = data.train.images
    y_train = data.train.labels
    x_test = data.test.images
    y_test = data.test.labels
    if raw:
        test = (dict(image=np.reshape(np.asarray(x_train, dtype=np.float32), newshape=[-1] + [64, 64, 1])),
                np.asarray(y_train, dtype=np.int32))
        train = (dict(image=np.reshape(np.asarray(x_test, dtype=np.float32), newshape=[-1] + [64, 64, 1])),
                 np.asarray(y_test, dtype=np.int32))
    else:
        test, train = as_image_dataset(x_test, x_train, y_test, y_train, image_shape=[64, 64, 1])
    data_format = 'channels_last'
    num_classes = 2
    return train, test, data_format, num_classes


def as_image_dataset(x_test, x_train, y_test, y_train, image_shape):
    train = tf.data.Dataset.from_tensor_slices((dict(image=
                                                     np.reshape(np.asarray(x_train, dtype=np.float32),
                                                                newshape=[-1] + image_shape)),
                                                np.asarray(y_train, dtype=np.int32)))
    test = tf.data.Dataset.from_tensor_slices((dict(image=
                                                    np.reshape(np.asarray(x_test, dtype=np.float32),
                                                               newshape=[-1] + image_shape)),
                                               np.asarray(y_test, dtype=np.int32)))
    return test, train


def loaderDictionary(datasetName):
    switcher = {
        "MNIST": loadMNIST,
        "Wheels": loadWheels,
        "GTSRB48x48": loadGTSRB48x48,
    }
    return switcher.get(datasetName, "Invalid Data Set")


# Define data loaders #####################################

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


class Dataset:

    def __init__(self, dataset_name, lables_dtype=np.int32, use_data_api=True):
        # load the dataset according to the datasetName
        self.dataset_name = dataset_name
        if use_data_api:
            self.train, self.test, self.data_format, self.num_classes = loaderDictionary(dataset_name)(False)
        else:
            self.train, self.test, self.data_format, self.num_classes = loaderDictionary(dataset_name)(True)

    # Define the training inputs
    def get_train_inputs(self, batch_size, buffer_size, num_epochs):
        """Return the input function to get the training data.
        Args:
            batch_size (int): Batch size of training iterator that is returned
                              by the input function.
        Returns:
            (Input function, IteratorInitializerHook):
                - Function that returns (features, labels) when called.
                - Hook to initialise input iterator.
        """
        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            """Returns training set as Operations.
            Returns:
                (features, labels) Operations that iterate over the dataset
                on every evaluation
            """
            with tf.name_scope('Training_data'):
                # Get Mnist data
                images = self.train[0]['image']
                labels = self.train[1]
                # Define placeholders
                images_placeholder = tf.placeholder(
                    images.dtype, images.shape)
                labels_placeholder = tf.placeholder(
                    labels.dtype, labels.shape)
                # Build dataset iterator
                dataset = tf.data.Dataset.from_tensor_slices(
                    (images_placeholder, labels_placeholder))

                dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, num_epochs))
                dataset = dataset.batch(batch_size)
                iterator = dataset.make_initializable_iterator()
                next_example, next_label = iterator.get_next()
                # Set runhook to initialize iterator
                iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(
                        iterator.initializer,
                        feed_dict={images_placeholder: images,
                                   labels_placeholder: labels})
                # Return batched (features, labels)
                return dict(image=next_example), next_label

        # Return function and hook
        return train_inputs, iterator_initializer_hook
