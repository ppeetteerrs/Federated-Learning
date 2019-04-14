from model import KerasModel
from utils import load_dataset, load_dummy
from random import sample
import argparse
import os
import sys
import json
import tensorflow as tf

parser = argparse.ArgumentParser(description="Parse Client Arguments")
parser.add_argument("-i", "--id", metavar='Client ID', type=int, nargs="?",
                    dest='id', help='Client ID', default=1)
parser.add_argument("-w", "--weights", metavar='Weights File', type=str, nargs="?",
                    dest='weights_file', help='Weights File Name')
parser.add_argument("-n", "--name", metavar='Simulator Name', type=str, nargs="?",
                    dest='name', help='Name of the simulator run', default="default")
args = parser.parse_args()


class Client():
    def __init__(self):
        self.id = args.id
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_metrics = tf.keras.metrics.Mean(name='loss')
        self.acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

        # Generate the Keras Model
        dummy_data = load_dummy(args.name)
        self.model = KerasModel()
        self.model(dummy_data)
        self.model.load_weights(args.weights_file)
        self.dataset = load_dataset(args.name, self.id)
        self.datagen = iter(self.dataset)
        self.acc_gradient = None

    def iterate(self):
        # Iterate through all batches
        for batch in self.datagen:
            self.train_step(batch)
        print("Client {} results: Loss - {}, Acc - {}"
              .format(self.id, self.loss_metrics.result(), self.acc_metrics.result() * 100))

    def train_step(self, batch):
        # Calculate outcome for one batch
        with tf.GradientTape() as tape:
            predictions = self.model(batch["x"], training=True)
            loss = self.loss(batch["y"], predictions)
            grads = tape.gradient(loss, self.model.trainable_variables)

        # Accumulate gradients
        self.accumulate_gradients(grads)

        # Apply gradients to model
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_metrics(loss)
        self.acc_metrics(batch["y"], predictions)

    def accumulate_gradients(self, gradient: list):
        if self.acc_gradient is None:
            self.acc_gradient = gradient
        else:
            self.acc_gradient = [tf.add(old_grad, new_grad) for old_grad, new_grad in zip(self.acc_gradient, gradient)]


Client().iterate()
