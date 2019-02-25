import sys
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.abspath('..'))
from dataset import Dataset
import yaml

model_dir = '../trained/GTSRB48x48/f4185cdcdcf204f4deb72d92f076cdb13561e60c_1'
saved_model = '1548855071'
configFile = open("../config.yaml")
data = yaml.load(configFile)
dataset = Dataset(data.get("dataset"))

tf.reset_default_graph()

def input_eval():
    return sess.run(dataset.test.batch(1000).make_one_shot_iterator().get_next())


with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, ['serve'], model_dir + '_quantized/' + saved_model)
    saver = tf.train.import_meta_graph(meta_graph)
    sess.run(tf.initializers.global_variables())
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    all_vars = tf.get_default_graph().as_graph_def().node
    print('Possible Variables to Plot')
    for i, var in enumerate(all_vars):
        print('{} : '.format(i) + var.name)

    sel = input('Select Variables to Plot: ')
    selected_var = sess.graph.get_tensor_by_name(all_vars[int(sel)].name + ':0')
    X = sess.graph.get_tensor_by_name('Reshape:0')
    x, y = input_eval()
    output = sess.run(selected_var, feed_dict={X: x['image']})
    plt.hist(np.reshape(output, [-1, 1]), bins=40)
    plt.show()
