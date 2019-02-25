from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import numpy
import os
import collections
import matplotlib.pyplot as plt
import tensorflow as tf


def CNN_ImageDataHandler(train_dir,
                         one_hot=False,
                         dtype=numpy.float32,
                         reshape=True,
                         validation_size=100,
                         distortBatch=False,
                         onlyTestSet=False):
    TRAIN_IMAGES = 'train_images.idx3-ubyte.gz'
    TRAIN_LABELS = 'train_labels.idx1-ubyte.gz'
    TEST_IMAGES = 'test_images.idx3-ubyte.gz'
    TEST_LABELS = 'test_labels.idx1-ubyte.gz'

    if onlyTestSet == False:
        local_file = os.path.join(train_dir, TRAIN_IMAGES)
        with open(local_file, 'rb') as f:
            train_images = extract_images(f)

        local_file = os.path.join(train_dir, TRAIN_LABELS)
        with open(local_file, 'rb') as f:
            train_labels = extract_labels(f, one_hot=one_hot)

        if not 0 <= validation_size <= len(train_images):
            raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'
                    .format(len(train_images), validation_size))

        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape, distortBatch=distortBatch)
        validation = DataSet(validation_images,
                             validation_labels,
                             dtype=dtype,
                             reshape=reshape)

    local_file = os.path.join(train_dir, TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)
    local_file = os.path.join(train_dir, TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    if onlyTestSet == False:
        Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
        return Datasets(train=train, validation=validation, test=test)

    if onlyTestSet == True:
        Datasets = collections.namedtuple('Datasets', ['test'])
        return Datasets(test=test)


class DataSet:

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=numpy.float32,
                 reshape=True,
                 distortBatch=False):
        """Construct a DataSet.
        `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (numpy.uint8, numpy.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self.__reshape = reshape
        self.__distortBatch = distortBatch
        self.__num_examples = images.shape[0]
        self.__numOfRows = images.shape[1]
        self.__numOfCols = images.shape[2]
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    self.__numOfRows * self.__numOfCols)
        if dtype == numpy.float32:
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self.__images = images
        self.__labels = labels
        self.__epochs_completed = 0
        self.__index_in_epoch = 0
        self.__distortionFactorHigh = 2.0
        self.__distortionFactorLow = 0.25
        self.__distortionBiasHigh = 0.5
        self.__distortionBiasLow = -0.1

    @property
    def images(self):
        return self.__images

    @property
    def labels(self):
        return self.__labels

    @property
    def num_examples(self):
        return self.__num_examples

    @property
    def numOfRows(self):
        return self.__numOfRows

    @property
    def numOfCols(self):
        return self.__numOfCols

    @property
    def epochs_completed(self):
        return self.__epochs_completed

    @property
    def index_in_epoch(self):
        return self.__index_in_epoch

    @index_in_epoch.setter
    def index_in_epoch(self, index):
        self.__index_in_epoch = index

    @property
    def distortionFactorHigh(self):
        return self.__distortionFactorHigh

    @property
    def distortionFactorLow(self):
        return self.__distortionFactorLow

    @property
    def distortionBiasHigh(self):
        return self.__distortionBiasHigh

    @property
    def distortionBiasLow(self):
        return self.__distortionBiasLow

    @distortionFactorHigh.setter
    def distortionFactorHigh(self, factor):
        self.__distortionFactorHigh = factor

    @distortionFactorLow.setter
    def distortionFactorLow(self, factor):
        self.__distortionFactorLow = factor

    @distortionBiasHigh.setter
    def distortionBiasHigh(self, bias):
        self.__distortionBiasHigh = bias

    @distortionBiasLow.setter
    def distortionBiasLow(self, bias):
        self.__distortionBiasLow = bias

    @property
    def distortedBatch(self):
        return self.__distortBatch

    @property
    def coverPart(self):
        return self.__coverPart

    @coverPart.setter
    def coverpart(self, part):
        self.__coverPart = part

    def shuffle(self):
        """Shuffles images and labels"""
        perm = numpy.arange(self.__num_examples)
        numpy.random.shuffle(perm)
        self.__images = self.__images[perm]
        self.__labels = self.__labels[perm]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.__index_in_epoch
        self.__index_in_epoch += batch_size
        if self.__index_in_epoch > self.__num_examples:
            # Finished epoch
            self.__epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self.__num_examples)
            numpy.random.shuffle(perm)
            self.__images = self.__images[perm]
            self.__labels = self.__labels[perm]
            # Start next epoch
            start = 0
            self.__index_in_epoch = batch_size
            assert batch_size <= self.__num_examples
        end = self.__index_in_epoch
        if self.__distortBatch == True:
            return self.__distortImages(self.__images[start:end].copy()), self.__labels[start:end]
        else:
            return self.__images[start:end], self.__labels[start:end]

    def __distortImages(self, images):
        """This function loops over all images (input) and applies function __distortSingleImage (to each image)

        Args:
            images:   If self.__reshape == True: 2D array [numOfImages, sizeRow * sizeCol]
                      If self.__reshape == False: 3D array [numOfImages, sizeRow, sizeCol]

        Return:
            images:   If self.__reshape == True: 2D array [numOfImages, sizeRow * sizeCol]
                      If self.__reshape == False: 3D array [numOfImages, sizeRow, sizeCol]
        """
        if self.__reshape == False:
            images = tf.reshape(images, [-1, self.__numOfRow * self.__numOfCols])
        numOfImages = images.shape[0]
        # flip randomly (p=0.5) images
        flipId = numpy.random.randint(0, 2, numOfImages, bool)

        flipPixelId = numpy.reshape(numpy.flip(
            numpy.reshape(numpy.arange(self.__numOfRows * self.__numOfCols), [self.__numOfRows, self.__numOfCols]),
            axis=1), [self.__numOfRows * self.__numOfCols])
        images[flipId] = numpy.transpose(numpy.transpose(images[flipId])[flipPixelId])

        # change of intensity with contrast or brightness
        brightnessId = numpy.random.randint(0, 2, numOfImages, bool)
        contrastId = brightnessId == False
        numOfBrightnessChange = numpy.sum(brightnessId)
        numOfContrastChange = numpy.sum(contrastId)

        BrightnesssChange = [rand * self.__distortionBiasHigh if rand > 0 else - rand * self.__distortionBiasLow for
                             rand in numpy.random.uniform(-1, 1, numOfBrightnessChange)]
        ContrastChange = [
            rand * (self.__distortionFactorHigh - 1) + 1 if rand > 0 else (rand * (1 - self.__distortionFactorLow) + 1)
            for rand in numpy.random.uniform(-1, 1, numOfContrastChange)]

        # Brightness
        images[brightnessId] = numpy.transpose(numpy.transpose(images[brightnessId]) + BrightnesssChange)
        images[contrastId] = numpy.transpose(numpy.transpose(images[contrastId]) * ContrastChange)

        # clip
        images[images > 1] = 1.0
        images[images < 0] = 0.0

        if self.__reshape == False:
            images = numpy.reshape(images, -1, self.__numOrRows, self.__numOfCols)

        # images = tf.map_fn(lambda image: self.__distortSingleImage(image), images)
        return images

    def showPanoramaLoadedImage(self):
        """plots the panorama image of the loaded images"""

        if self.__reshape == False:
            images = self.__images.reshape(self.__images.shape[0], self.__numOfRows * self.__numOfCols)
        else:
            images = self.__images

        showPanoramaImage(images, self.__numOfRows, self.__numOfCols)


def showPanoramaImage(images, numOfRows, numOfCols):
    """This function plots the panorama image

    Args:
        images:\t\t\t       2D array [numOfImages, sizeRow * sizeCol]
        numOfRows:    Number of rows in per image
        numOfCols:    Number od columns per image

    Return:
        image:        2D array (panorama image)

    """
    numOfImagesPerSide = (numpy.ceil(numpy.sqrt(images.shape[0]))).astype(numpy.int)
    numOfRestImages = numOfImagesPerSide ** 2 - images.shape[0]

    restImages = numpy.zeros([numOfRestImages, images.shape[1]], dtype=numpy.float32)
    imagesPanorama = numpy.concatenate((images, restImages))
    imagesPanorama = numpy.reshape(imagesPanorama, [numOfImagesPerSide ** 2, numOfRows, numOfCols])
    imagesPanorama = numpy.transpose(imagesPanorama, [0, 2, 1])
    imagesPanorama = numpy.reshape(imagesPanorama, [numOfImagesPerSide, numOfCols * numOfImagesPerSide, numOfRows])
    imagesPanorama = numpy.transpose(imagesPanorama, [0, 2, 1])
    imagesPanorama = numpy.reshape(imagesPanorama, [numOfRows * numOfImagesPerSide, numOfRows * numOfImagesPerSide])
    plt.figure()
    plt.imshow(imagesPanorama, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        num_classes = len(numpy.unique(labels))
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels
