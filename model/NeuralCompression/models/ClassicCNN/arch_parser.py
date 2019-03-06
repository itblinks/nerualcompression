import tensorflow as tf
import ast


def readIDintoDict(fileHandle, ID):
    for i, line in enumerate(fileHandle):
        if i == ID:
            rawID = line

    raw_splitted = rawID.split('___')
    layer_list = list()
    for i, raw_layer in enumerate(raw_splitted):
        if i != 0: #ignore first line, this only contains the ID
            # separate at the first { where the type ends. Add a {, since it is removed by the split method
            layer_list.append(ast.literal_eval('{' + raw_layer.split('{')[1]))
            layer_list[i-1]['Type'] = raw_layer.split('{')[0] # Add the type back to the dictionary


if __name__ == '__main__':
    file = open('../architectures/arch_log.txt')
    ID = 505
    readIDintoDict(file, ID)
