
import matplotlib
import numpy as np
from utils import load_h5, load_labels, KerasModel, separate_data_by_class, generate_client_datasets
import os
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.python.keras.optimizer_v2 import adam

tf.compat.v1.enable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

nest = tf.contrib.framework.nest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

rawX, rawY = load_h5("cifar/train_data.h5")
labels = load_labels("cifar/labels.h5")
separated_data = separate_data_by_class(rawX, rawY, labels)
datasets = generate_client_datasets(separated_data, 2, 64, 30, 10, 50, 1, 4, True)
sample = nest.map_structure(
    lambda x: x.numpy(), iter(datasets[0]).next()
)


def model_fn():

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))

    model = KerasModel()
    model.compile(
        optimizer=adam.Adam(),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return tff.learning.from_compiled_keras_model(model, sample)


iterative_process = tff.learning.build_federated_averaging_process(model_fn)
str(iterative_process.initialize.type_signature)
state = iterative_process.initialize()
state, metrics = iterative_process.next(state, [datasets[0]])
print('round  1, metrics={}'.format(metrics))

for round_num in range(1, 11):
    state, metrics = iterative_process.next(state, datasets[0:4])
    print('round {:2d}, metrics={}'.format(round_num, metrics))
