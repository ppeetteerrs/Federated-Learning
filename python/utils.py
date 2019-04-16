# %%
import pickle
import os
import numpy as np
from collections import OrderedDict
from random import sample, choice
import tensorflow as tf
import json


# Loads h5 data
def load_h5(file_path):
    with open(file_path, "rb") as data_file:
        X, Y = pickle.load(data_file)
        X = X / 255
    return X.astype(np.float32), Y.astype(np.int32)


# Load labels
def load_labels(file_path):
    with open(file_path, "rb") as labels_file:
        labels = pickle.load(labels_file)
    return labels


# Load dataset
def load_dataset(directory: str, client_id: int):
    data_file_path = os.path.join("temp", directory, "data_client_{}.h5".format(client_id))
    with open(data_file_path, "rb") as dataset_file:
        dataset = pickle.load(dataset_file)
    return (tf.data.Dataset
            .from_tensor_slices((dataset["x"], dataset["y"]))
            .repeat(dataset["epochs"])
            .map(lambda x, y: OrderedDict([("x", x), ("y", y)]))
            .shuffle(len(dataset["y"]) + 1)
            .batch(dataset["batch_size"]))


# Load test dataset
def load_test_dataset(file_path: str):
    testX, testY = load_h5(file_path)
    return (tf.data.Dataset
            .from_tensor_slices((testX, testY))
            .map(lambda x, y: OrderedDict([("x", x), ("y", y)]))).batch(64)


# Load dummy
def load_dummy(directory: str):
    dummy_file_path = os.path.join("temp", directory, "dummy_data.h5")
    with open(dummy_file_path, "rb") as dummy_file:
        dummy_data = pickle.load(dummy_file)
    return tf.convert_to_tensor(dummy_data)


# Load Client Gradient
def load_gradient(directory: str, step: int, client_id: int):
    # gradient_path = os.path.join("temp", directory, "gradient_step_{}_client_{}.h5".format(step, client_id))
    gradient_path = os.path.join("temp", directory, "gradient_client_{}.h5".format(client_id))
    with open(gradient_path, "rb") as gradient_file:
        gradient = pickle.load(gradient_file)
    return gradient


# Load Client Gradients
def load_gradients(directory: str, step: int, client_ids: int):
    gradients = [load_gradient(directory, step, client_id) for client_id in client_ids]
    average_gradient = np.sum(gradients, axis=0) / len(client_ids)
    return average_gradient


# Separate training data by class
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


# Creates dataset files for clients
# Arguments: number of clients, minimum samples per client, maximum samples per client,
# minimum no. of classes per client, maximum no. of classes per client
def generate_client_dataset_files(
    dataset: {},
    directory: str,
    epochs: int,
    batch_size: int,
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
            "y": [],
            "epochs": epochs,
            "batch_size": batch_size,
            "n_samples": n_samples
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
        data_folder_path = os.path.join("temp", directory)
        data_file_path = os.path.join("temp", directory, "data_client_{}.h5".format(client_id))
        os.makedirs(data_folder_path, exist_ok=True)
        with open(data_file_path, "wb") as file:
            pickle.dump(client_data, file)
        if client_id == 0:
            dummy_file_path = os.path.join("temp", directory, "dummy_data.h5")
            with open(dummy_file_path, "wb") as file:
                pickle.dump(client_data["x"], file)
    for i in range(n_clients):
        generate_client_dataset(
            i,
            choose(n_classes_min, n_classes_max),
            choose(n_samples_min, n_samples_max),
        )
