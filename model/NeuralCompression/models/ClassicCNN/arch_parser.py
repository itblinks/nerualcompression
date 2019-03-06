import tensorflow as tf
import json


def readIDintoDict(fileHandle, ID):
    for i, line in enumerate(fileHandle):
        if i == ID:
            rawID = line

    raw_splitted = rawID.split('___')
    layer_list = list()
    for i, raw_layer in enumerate(raw_splitted):
        if i != 0:
            layer_list.append(json.loads(raw_layer))


if __name__ == '__main__':
    file = open('../architectures/arch_log.txt')
    ID = 505
    readIDintoDict(file, ID)
