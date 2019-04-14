# %%
import pickle
import numpy as np
from collections import OrderedDict
import warnings
from random import sample, choice
import matplotlib
import tensorflow as tf
import tensorflow_federated as tff

layers = tf.keras.layers

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

tf.compat.v1.enable_v2_behavior()
tf.enable_eager_execution(config=config)


# Loads h5 data
def load_h5(file_path):
    with open(file_path, "rb") as data_file:
        X, Y = pickle.load(data_file)
        X = X / 255
    return X.astype(np.float32)[:3000], Y.astype(np.int32)[:3000]


# Load labels
def load_labels(file_path):
    with open(file_path, "rb") as data_file:
        labels = pickle.load(data_file)
    return labels


# Separate data by class
def separate_data_by_class(X: [], Y: [], labels: [] = None):
    output = {}
    for x_item, y_item in zip(X, Y):
        name = str(y_item) if labels is None else labels[y_item]
        if name in output:
            output[name]["x"].append(x_item)
            output[name]["y"].append(y_item)
        else:
            output[name] = {}
            output[name]["x"] = [x_item]
            output[name]["y"] = [y_item]
    return output


# Creates dataset for clients
# Arguments: number of clients, minimum samples per client, maximum samples per client,
# minimum no. of classes per client, maximum no. of classes per client
def generate_client_datasets(
    dataset: {},
    epochs: int = 1,
    batch_size: int = 32,
    n_clients: int = 1,
    n_samples_min: int = 100,
    n_samples_max: int = None,
    n_classes_min: int = 1,
    n_classes_max: int = None,
    no_repeat: bool = False,
):
    classes = dataset.keys()

    # Choose a value between min and max
    def choose(min_val, max_val):
        if max_val is None:
            return min_val
        else:
            return choice(range(min_val, max_val + 1))

    def generate_client_dataset(client_id: int, n_classes: int, n_samples: int):
        client_data = {
            "x": [],
            "y": []
        }
        chosen_classes = sample(classes, n_classes)
        chosen_indices = [0] + \
            sample(range(n_samples), n_classes - 1) + [n_samples]
        chosen_indices.sort()
        for i, class_label in enumerate(chosen_classes):
            data = dataset[class_label]
            sample_no = chosen_indices[i + 1] - chosen_indices[i]
            samples_left = len(data["y"])

            if sample_no > samples_left:
                # Not enough data remaining
                print("Not enough samples for client", client_id)
                sample_no = samples_left
            selected_indices = sorted(sample(range(samples_left), sample_no), reverse=True)
            for j in selected_indices:
                client_data["x"].append(data["x"][j])
                client_data["y"].append([data["y"][j]])
            if no_repeat:
                for j in selected_indices:
                    del data["x"][j]
                    del data["y"][j]
        client_data["x"] = np.asarray(client_data["x"])
        client_data["y"] = np.asarray(client_data["y"])
        return (
            tf.data.Dataset.from_tensor_slices(
                (client_data["x"], client_data["y"]))
            .repeat(epochs)
            .map(lambda x, y: OrderedDict([("x", tf.reshape(x, [-1])), ("y", y)]))
            .shuffle(len(client_data["y"]))
            .batch(batch_size)
        )

    return [
        generate_client_dataset(
            i,
            choose(n_classes_min, n_classes_max),
            choose(n_samples_min, n_samples_max),
        ) for i in range(n_clients)
    ]


# Defines model
class KerasModel(tf.keras.Model):
    def __init__(self):
        super(KerasModel, self).__init__()
        self.reshape = layers.Reshape((32, 32, 3), input_shape=(3072,))
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3))
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(32, (3, 3))
        self.relu2 = layers.ReLU()
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(rate=0.25)

        self.conv3 = layers.Conv2D(64, (3, 3), padding='same')
        self.relu3 = layers.ReLU()
        self.conv4 = layers.Conv2D(64, (3, 3))
        self.relu4 = layers.ReLU()
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2))
        self.dropout2 = layers.Dropout(rate=0.25)

        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(512)
        self.relu5 = layers.ReLU()
        self.dropout3 = layers.Dropout(rate=0.5)
        self.dense2 = layers.Dense(10)
        self.softmax = layers.Softmax()

    def call(self, inputs, training: bool = True):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return self.softmax(x)


# Defines FL model
class FLModel(tff.learning.TrainableModel):
    def __init__(self):
        pass
