import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import hashlib
from dataset import Dataset
from models.ClassicCNN.classiccnnmodel import Model as CNNModel
import pandas as pd
import csv

batch_size = 32
buffer_size = 180000
num_epochs = 40
train = False
eval_quant = True
attempt_ID = 1  # Attempt ID number for multiple passes with the same network


def main():
    # import yaml config file to set up network
    configFile = open("config.yaml")
    data = yaml.load(configFile)
    dataset = Dataset(data.get("dataset"), use_data_api=False)
    data_format = dataset.data_format
    num_classes = dataset.num_classes
    parseID = data.get("model").get("parseID")
    ID = data.get("model").get("ID")
    arch_file = data.get("model").get("arch_file")
    conv_size = data.get("model").get("conv_size")
    kernel_size = data.get("model").get("kernel_size")
    num_filter = data.get("model").get("num_filter")
    pool_size = data.get("model").get("pool_size")
    pool_stride = data.get("model").get("pool_stride")
    conv_stride = data.get("model").get("conv_stride")
    dense_depth = data.get("model").get("dense_depth")
    dense_neurons = data.get("model").get("dense_neurons")
    dropout_factor = data.get("model").get("dropout_factor")
    quant_w_params = data.get('quantization').get('quant_w_params')
    quant_act_format = data.get("quantization").get("quant_act_format")

    # print a hash of the configuration
    hashfun = hashlib.sha1()
    if parseID:
        with open(arch_file) as arch_file_txt:
            for i, line in enumerate(arch_file_txt):
                if i == ID:
                    hashfun.update(str(line).encode('utf-8'))

    else:
        for i, line in enumerate(data.get("model")):
            if line != "parseID" and line != "ID" and line != "arch_file":
                hashfun.update(str(data.get("model").get(line)).encode("utf-8"))
        hashfun.update(str(data.get("dataset")).encode("utf-8"))

    digest = hashfun.hexdigest()
    print('The model-config has hash: {}'.format(digest))

    model = CNNModel(conv_size=conv_size, kernel_size=kernel_size, num_filter=num_filter, pool_size=pool_size,
                     pool_stride=pool_stride,
                     conv_stride=conv_stride, data_format=data_format, dense_depth=dense_depth,
                     dense_neurons=dense_neurons, dropout=dropout_factor, num_classes=num_classes,
                     quant_act_format=quant_act_format, ID=ID, parseID=parseID, arch_file=arch_file)

    feature_column = tf.feature_column.numeric_column(key='image',
                                                      shape=dataset.train[0]['image'].shape[1:])
    my_checkpointing_config = tf.estimator.RunConfig(
        keep_checkpoint_max=2,  # Retain the 5 most recent checkpoints.
        save_checkpoints_steps=1240
    )
    model_dir = 'trained/' + dataset.dataset_name + '/' + digest + '_' + str(attempt_ID)

    tf.logging.set_verbosity(tf.logging.INFO)

    input_eval = tf.estimator.inputs.numpy_input_fn(*dataset.test, batch_size=32, num_epochs=1, shuffle=False,
                                                    queue_capacity=20000, num_threads=1)
    input_pred = tf.estimator.inputs.numpy_input_fn(*dataset.test, batch_size=1, num_epochs=1, shuffle=True,
                                                    queue_capacity=20000, num_threads=1)

    if train:
        estimator = tf.estimator.Estimator(model_fn=model.classiccnn_model_fn,
                                           params={'learning_rate': 0.001, 'feature_columns': feature_column,
                                                   'quant_act': False, 'decay': True, 'decay_rate': 0.1,
                                                   'decay_steps': 12800},
                                           config=my_checkpointing_config, model_dir=model_dir)

        input_train, train_input_hook = dataset.get_train_inputs(batch_size=batch_size, buffer_size=buffer_size,
                                                                 num_epochs=num_epochs)
        train_specs = tf.estimator.TrainSpec(input_fn=input_train, hooks=[train_input_hook])
        eval_specs = tf.estimator.EvalSpec(input_fn=input_eval, throttle_secs=10)
        tf.estimator.train_and_evaluate(estimator, train_specs, eval_specs)

    estimator = tf.estimator.Estimator(model_fn=model.classiccnn_model_fn,
                                       params={'learning_rate': 0.001, 'feature_columns': feature_column,
                                               'quant_act': False},
                                       config=my_checkpointing_config, model_dir=model_dir)
    # Evaluate the model
    evaluation_pre = estimator.evaluate(input_eval)
    num_quant_exp = np.size(quant_w_params.get('quant_conv'))

    if eval_quant:
        # create log file
        evaluation_after = {}
        csv_fieldnames = ['quant_conv', 'quant_dense', 'num_bin_filt_conv', 'max_num_bin_filt', 'num_bin_filt_dense',
                          'paper_approach', 'max_opt_runs', 'pow_of_two', 'bin_quant', 'log_quant', 'accuracy', 'loss',
                          'global_step']
        csv_file = open(model_dir + '/results.csv', 'w')
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        csv_writer.writerow(evaluation_pre)
        for i in range(num_quant_exp):
            q_param = {'quant_conv': quant_w_params.get('quant_conv')[i],
                       'quant_dense': quant_w_params.get('quant_dense')[i],
                       'num_bin_filt_conv': quant_w_params.get('num_bin_filt_conv')[i],
                       'max_num_bin_filt': quant_w_params.get('max_num_bin_filt')[i],
                       'num_bin_filt_dense': quant_w_params.get('num_bin_filt_dense')[i],
                       'paper_approach': quant_w_params.get('paper_approach')[i],
                       'max_opt_runs': quant_w_params.get('max_opt_runs')[i],
                       'pow_of_two': quant_w_params.get('pow_of_two')[i],
                       'bin_quant': quant_w_params.get('bin_quant')[i],
                       'log_quant': quant_w_params.get('log_quant')[i]
                       }
            model.quantize_weights(model_dir, quant_param=q_param)

            estimator = tf.estimator.Estimator(model_fn=model.classiccnn_model_fn,
                                               params={'learning_rate': 0.001, 'feature_columns': feature_column,
                                                       'quant_act': True},
                                               config=my_checkpointing_config, model_dir=model_dir + '_quantized')
            q_param.update(estimator.evaluate(input_eval))
            csv_writer.writerow(q_param)
        csv_file.flush()
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        {'image': tf.FixedLenFeature(shape=dataset.train[0]['image'].shape, dtype=tf.float32)})
    estimator.export_saved_model(model_dir + '_quantized', serving_input_fn)

    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    template = ('\n"{}"({:.1f}%)')
    prediction = estimator.predict(input_fn=input_pred)

    annotations = '../../dataset/GTSRB48x48/signnames.csv'

    classes = pd.read_csv(annotations)
    class_names = {}
    for i, row in classes.iterrows():
        class_names[str(row[0])] = row[1]

    for pred_dict, i in zip(prediction, range(columns * rows)):
        class_id = pred_dict['classes']
        probability = np.max(pred_dict['probabilities'])
        fig.add_subplot(rows, columns, i + 1)
        image = plt.imshow(np.reshape(pred_dict['image'], [48, 48, -1]))
        image.axes.get_xaxis().set_visible(False)
        image.axes.get_yaxis().set_visible(False)
        plt.title(template.format(str(class_id), 100 * probability))
    plt.tight_layout()
    plt.show()

    print('Evaluation before Quantization: {}'.format(evaluation_pre))


if __name__ == '__main__':
    main()
